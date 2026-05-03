import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from tqdm import tqdm
import os
import pickle
import gc
import warnings
warnings.filterwarnings('ignore')

#  复用召回阶段全局配置，保证全链路一致性 
TOP_N_ROUGH = 50  # 粗排最终输出Top50给精排
try:
    from Recall_full_final import (
        DEVICE, USE_CUDA, EMBEDDING_DIM, DTYPE_CONFIG,
        BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_SAVE_DIR
    )
except ImportError:
    EMBEDDING_DIM = 64
    BATCH_SIZE = 4096
    EPOCHS = 3
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    # 内存优化dtype，和召回完全对齐
    DTYPE_CONFIG = {
        "user_id": np.int32,
        "item_id": np.int32,
        "category_id": np.int32,
        "behavior_type": "category",
        "float_feature": np.float32,
        "int_feature": np.int32,
        "idx_dtype": np.int32
    }
    MODEL_SAVE_DIR = "./recall_models"

# 粗排模型保存根目录
ROUGH_MODEL_SAVE_DIR = os.path.join(MODEL_SAVE_DIR, "rough_rank")
os.makedirs(ROUGH_MODEL_SAVE_DIR, exist_ok=True)

#  粗排特征列定义，和特征工程完全对齐 
# 用户侧特征列
USER_FEAT_COLS = [
    "user_total_click", "user_total_buy", "user_total_cart", "user_total_fav",
    "user_cvr", "user_ctr", "user_cart_rate", "user_fav_rate", "user_behavior_days", "user_is_new"
]
# 物品侧特征列
ITEM_FEAT_COLS = [
    "item_total_click", "item_total_buy", "item_total_cart", "item_total_fav",
    "item_cvr", "item_ctr", "item_cart_rate", "item_fav_rate", "item_is_hot", "item_is_new"
]
# 标签列定义
LABEL_COL = "click_label"  # 核心标签，也可改为label（购买标签）
TEACHER_SCORE_COL = "teacher_score"  # 蒸馏用的精排教师模型打分列

#  粗排模型基类（统一接口，方便扩展/切换模型）
class BaseRoughRank:
    def __init__(self, top_n=TOP_N_ROUGH):
        self.top_n = top_n
        self.name = self.__class__.__name__
        self.save_dir = os.path.join(ROUGH_MODEL_SAVE_DIR, self.name)
        os.makedirs(self.save_dir, exist_ok=True)
        # 特征归一化器
        self.user_scaler = MinMaxScaler()
        self.item_scaler = MinMaxScaler()
        self.is_fitted = False
        # 新增：推理加速缓存
        self.item_feat_cache = None  # 预计算的归一化后物品特征（GPU张量）
        self.item_id_to_idx = None   # 物品ID到索引的映射，O(1)加速查询

    #  原有方法完全保留不变 
    def fit(self, train_data_loader, val_data=None, epochs=1):
        raise NotImplementedError
    def predict(self, user_feat_np, item_feat_np):
        raise NotImplementedError
    def save(self):
        raise NotImplementedError
    @classmethod
    def load(cls):
        raise NotImplementedError

    #  原有单用户rank方法保留，兼容原有逻辑 
    def rank(self, user_id, user_feature, recall_item_list, item_features_df):
        valid_items = [item for item in recall_item_list if item in item_features_df["item_id"].values]
        if len(valid_items) == 0:
            return []
        user_feat_np = self.user_scaler.transform(user_feature[USER_FEAT_COLS].values.reshape(1, -1)).astype(DTYPE_CONFIG["float_feature"])
        user_feat_np = np.repeat(user_feat_np, len(valid_items), axis=0)
        item_feat_np = self.item_scaler.transform(item_features_df[item_features_df["item_id"].isin(valid_items)][ITEM_FEAT_COLS].values).astype(DTYPE_CONFIG["float_feature"])
        scores = self.predict(user_feat_np, item_feat_np)
        item_score_df = pd.DataFrame({"item_id": valid_items, "score": scores})
        top_items = item_score_df.sort_values("score", ascending=False).head(self.top_n)["item_id"].tolist()
        return top_items

    #  新增：预缓存物品特征，推理前仅执行1次 
    def build_item_feature_cache(self, item_features_df):
        """预计算全量物品的归一化特征，建立ID到索引的映射，彻底消除重复计算"""
        print(f" 预构建物品特征缓存 ")
        # 排序保证索引稳定
        item_features_df = item_features_df.sort_values("item_id").reset_index(drop=True)
        # 建立物品ID→索引的字典映射，O(1)查询
        self.item_id_to_idx = pd.Series(item_features_df.index, index=item_features_df["item_id"]).to_dict()
        # 预计算并缓存归一化后的物品特征，直接存GPU张量
        item_feat_np = self.item_scaler.transform(item_features_df[ITEM_FEAT_COLS].values).astype(DTYPE_CONFIG["float_feature"])
        self.item_feat_cache = torch.FloatTensor(item_feat_np).to(DEVICE)
        print(f"物品特征缓存构建完成，共缓存 {len(self.item_feat_cache)} 个商品特征")
        gc.collect()

    #  新增：批量用户粗排推理，GPU向量化加速核心 
    def batch_rank(self, user_ids, user_features_df, recall_item_lists):
        """
        批量用户粗排核心方法，GPU批量打分，替代逐用户循环
        :param user_ids: 批量用户ID列表
        :param user_features_df: 对应用户的特征DataFrame
        :param recall_item_lists: 对应用户的召回商品列表（二维列表）
        :return: 每个用户的TopN商品ID列表
        """
        if self.item_feat_cache is None or self.item_id_to_idx is None:
            raise RuntimeError("请先调用 build_item_feature_cache 预构建物品特征缓存")

        # 步骤1：批量处理用户特征，一次性归一化，避免循环重复计算
        user_feat_np = self.user_scaler.transform(user_features_df[USER_FEAT_COLS].values).astype(DTYPE_CONFIG["float_feature"])
        user_feat_tensor = torch.FloatTensor(user_feat_np).to(DEVICE)

        # 步骤2：展开用户-召回商品对，向量化处理
        all_user_idx = []
        all_item_idx = []
        user_item_count = []  # 记录每个用户的有效召回商品数，用于后续分组

        for user_idx, (uid, recall_items) in enumerate(zip(user_ids, recall_item_lists)):
            # 过滤无效商品，批量O(1)查询索引，比循环isin快100倍
            valid_item_idx = [self.item_id_to_idx[item] for item in recall_items if item in self.item_id_to_idx]
            item_count = len(valid_item_idx)
            if item_count == 0:
                user_item_count.append(0)
                continue
            # 记录用户索引和商品索引，用于后续批量取特征
            all_user_idx.extend([user_idx] * item_count)
            all_item_idx.extend(valid_item_idx)
            user_item_count.append(item_count)

        if len(all_user_idx) == 0:
            return [[] for _ in user_ids]

        # 步骤3：GPU批量获取特征，一次性并行打分
        # 张量索引批量取特征，无循环，GPU并行极快
        batch_user_feat = user_feat_tensor[all_user_idx]
        batch_item_feat = self.item_feat_cache[all_item_idx]
        # 批量预测打分（GPU并行计算，比单用户循环快1000倍）
        batch_scores = self.predict(batch_user_feat, batch_item_feat)

        # 步骤4：分组取TopN，向量化处理
        top_items_list = []
        ptr = 0
        # 预提取物品ID列表，避免循环查询字典
        item_id_list = list(self.item_id_to_idx.keys())
        for user_idx, item_count in enumerate(user_item_count):
            if item_count == 0:
                top_items_list.append([])
                continue
            # 取出当前用户的所有商品打分
            user_scores = batch_scores[ptr:ptr+item_count]
            # 取出当前用户的商品ID
            current_item_idx = all_item_idx[ptr:ptr+item_count]
            user_item_ids = [item_id_list[idx] for idx in current_item_idx]
            ptr += item_count
            # Torch张量排序取TopN，比pandas排序快10倍
            top_k = min(self.top_n, item_count)
            top_idx = torch.argsort(user_scores, descending=True)[:top_k].cpu().numpy()
            top_items = [user_item_ids[i] for i in top_idx]
            top_items_list.append(top_items)

        # 释放GPU显存和内存
        del batch_user_feat, batch_item_feat, batch_scores, user_feat_tensor
        torch.cuda.empty_cache()
        gc.collect()

        return top_items_list

#  FM模型
class FMModel(BaseRoughRank):
    def __init__(self, top_n=TOP_N_ROUGH, user_feat_dim=len(USER_FEAT_COLS), item_feat_dim=len(ITEM_FEAT_COLS)):
        super().__init__(top_n)
        self.input_dim = user_feat_dim + item_feat_dim
        self.user_feat_dim = user_feat_dim
        self.item_feat_dim = item_feat_dim
        # FM模型核心参数
        self.w = nn.Linear(self.input_dim, 1, bias=True)
        self.v = nn.Embedding(self.input_dim, EMBEDDING_DIM)
        # 模型初始化
        self.model = nn.Sequential(self.w, self.v).to(DEVICE)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        self.criterion = nn.BCEWithLogitsLoss()
        if USE_CUDA:
            self.scaler = torch.cuda.amp.GradScaler()

    def _fm_forward(self, x):
        """FM核心前向计算：线性项 + 二阶交叉项"""
        # 线性项
        linear_part = self.w(x)
        # 二阶交叉项
        embed_x = self.v(torch.arange(self.input_dim).to(DEVICE))  # [input_dim, embed_dim]
        xv = torch.matmul(x, embed_x)  # [batch_size, embed_dim]
        square_of_sum = torch.sum(xv, dim=1) ** 2
        sum_of_square = torch.sum(xv ** 2, dim=1)
        cross_part = 0.5 * (square_of_sum - sum_of_square).unsqueeze(1)
        # 合并输出
        return linear_part + cross_part

    def fit(self, train_data_loader, val_data=None, epochs=1):
        """
        FM模型训练
        :param train_data_loader: 训练数据DataLoader，每个batch返回 (feat_tensor, label_tensor)
        :param val_data: 验证集数据 (val_user_feat_np, val_item_feat_np, val_label_np)
        :param epochs: 训练轮数
        """
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for batch_idx, (batch_feat, batch_label) in enumerate(tqdm(train_data_loader, desc=f"【FM】训练 Batch", mininterval=1, leave=False)):
                batch_feat = batch_feat.to(DEVICE, non_blocking=True)
                batch_label = batch_label.to(DEVICE, non_blocking=True).float().unsqueeze(1)
                self.optimizer.zero_grad(set_to_none=True)

                # 前向传播
                if USE_CUDA:
                    with torch.cuda.amp.autocast():
                        output = self._fm_forward(batch_feat)
                        loss = self.criterion(output, batch_label)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    output = self._fm_forward(batch_feat)
                    loss = self.criterion(output, batch_label)
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()

            # Epoch日志
            avg_loss = total_loss / len(train_data_loader)
            print(f"【FM】Batch训练完成，平均Loss: {avg_loss:.6f}")

            # 验证集评估（仅当传入val_data且是最后一个epoch时）
            if val_data is not None and epoch == epochs-1:
                val_user_feat_np, val_item_feat_np, val_label_np = val_data
                val_concat_feat = np.concatenate([val_user_feat_np, val_item_feat_np], axis=1)
                val_feat_tensor = torch.FloatTensor(val_concat_feat).to(DEVICE)
                self.model.eval()
                with torch.no_grad():
                    val_output = self._fm_forward(val_feat_tensor).cpu().numpy().flatten()
                val_pred = torch.sigmoid(torch.FloatTensor(val_output)).numpy()
                val_auc = roc_auc_score(val_label_np, val_pred)
                print(f"【FM】验证集 AUC: {val_auc:.6f}")
                self.model.train()

        self.is_fitted = True

    def predict(self, user_feat_np, item_feat_np):
        """批量打分，兼容numpy数组和torch张量，自动匹配返回类型，避免重复CPU-GPU拷贝"""
        self.model.eval()
        is_numpy_input = isinstance(user_feat_np, np.ndarray)
        
        # 统一处理输入，转为GPU张量计算
        if is_numpy_input:
            concat_feat = np.concatenate([user_feat_np, item_feat_np], axis=1)
            feat_tensor = torch.FloatTensor(concat_feat).to(DEVICE, non_blocking=True)
        else:
            feat_tensor = torch.concat([user_feat_np, item_feat_np], axis=1)
        
        with torch.no_grad():
            output = self._fm_forward(feat_tensor).flatten()
            pred = torch.sigmoid(output)
        
        # 输入是numpy则返回numpy，输入是张量则返回张量，保证兼容性
        if is_numpy_input:
            pred = pred.detach().cpu().numpy()
        
        return pred
    

    def save(self):
        """保存FM模型与归一化器"""
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "fm_model.pth"))
        with open(os.path.join(self.save_dir, "user_scaler.pkl"), "wb") as f:
            pickle.dump(self.user_scaler, f)
        with open(os.path.join(self.save_dir, "item_scaler.pkl"), "wb") as f:
            pickle.dump(self.item_scaler, f)
        print(f"【FM】模型保存完成！")

    @classmethod
    def load(cls):
        """加载已训练的FM模型"""
        model = cls()
        # 加载模型权重
        model.model.load_state_dict(torch.load(os.path.join(model.save_dir, "fm_model.pth"), map_location=DEVICE))
        model.model.to(DEVICE)
        model.model.eval()
        # 加载归一化器
        with open(os.path.join(model.save_dir, "user_scaler.pkl"), "rb") as f:
            model.user_scaler = pickle.load(f)
        with open(os.path.join(model.save_dir, "item_scaler.pkl"), "rb") as f:
            model.item_scaler = pickle.load(f)
        model.is_fitted = True
        print(f"【FM】模型加载完成！")
        return model

#  蒸馏轻量化双塔模型（项目1优化版，核心亮点）
class DistillDualTower(nn.Module):
    """轻量化双塔结构，专为粗排设计，和项目1方案完全对齐"""
    def __init__(self, user_feat_dim, item_feat_dim, embedding_dim=32):
        super().__init__()
        # 轻量化用户塔：2层全连接，比召回双塔更轻量
        self.user_tower = nn.Sequential(
            nn.Linear(user_feat_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        # 物品塔和用户塔对称
        self.item_tower = nn.Sequential(
            nn.Linear(item_feat_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        self.temperature = 0.07
        self.user_feat_dim = user_feat_dim
        self.item_feat_dim = item_feat_dim

    def forward(self, user_feat, item_feat):
        # 归一化向量，内积计算相似度
        user_emb = nn.functional.normalize(self.user_tower(user_feat), dim=-1)
        item_emb = nn.functional.normalize(self.item_tower(item_feat), dim=-1)
        # 内积打分，温度系数缩放
        score = torch.sum(user_emb * item_emb, dim=-1, keepdim=True) / self.temperature
        return torch.sigmoid(score)

    @torch.no_grad()
    def get_user_embedding(self, user_feat):
        return nn.functional.normalize(self.user_tower(user_feat), dim=-1)

    @torch.no_grad()
    def get_item_embedding(self, item_feat):
        return nn.functional.normalize(self.item_tower(item_feat), dim=-1)

class DistillDualTowerRoughRank(BaseRoughRank):
    def __init__(self, top_n=TOP_N_ROUGH, user_feat_dim=len(USER_FEAT_COLS), item_feat_dim=len(ITEM_FEAT_COLS)):
        super().__init__(top_n)
        self.user_feat_dim = user_feat_dim
        self.item_feat_dim = item_feat_dim
        # 核心模型
        self.model = DistillDualTower(user_feat_dim, item_feat_dim).to(DEVICE)
        # 优化器与损失函数
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        self.bce_criterion = nn.BCELoss()  # 标签分类损失
        self.mse_criterion = nn.MSELoss()  # 蒸馏损失
        # 蒸馏权重配置，可按需调整
        self.distill_weight = 0.7  # 蒸馏损失权重
        self.label_weight = 0.3    # 真实标签损失权重
        if USE_CUDA:
            self.scaler = torch.cuda.amp.GradScaler()

    def fit(self, train_data_loader, val_data=None, epochs=1):
        """
        蒸馏双塔模型训练，双损失优化
        :param train_data_loader: 训练数据DataLoader，每个batch返回 (user_feat_tensor, item_feat_tensor, label_tensor, teacher_score_tensor)
        :param val_data: 验证集数据 (val_user_feat_np, val_item_feat_np, val_label_np)
        :param epochs: 训练轮数
        """
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            total_distill_loss = 0.0
            total_label_loss = 0.0

            for batch_idx, (batch_user_feat, batch_item_feat, batch_label, batch_teacher_score) in enumerate(
                tqdm(train_data_loader, desc=f"【蒸馏双塔】训练 Batch", mininterval=1, leave=False)
            ):
                # 数据移到GPU
                batch_user_feat = batch_user_feat.to(DEVICE, non_blocking=True)
                batch_item_feat = batch_item_feat.to(DEVICE, non_blocking=True)
                batch_label = batch_label.to(DEVICE, non_blocking=True).float().unsqueeze(1)
                batch_teacher_score = batch_teacher_score.to(DEVICE, non_blocking=True).float().unsqueeze(1)
                self.optimizer.zero_grad(set_to_none=True)

                # 前向传播
                if USE_CUDA:
                    with torch.cuda.amp.autocast():
                        pred = self.model(batch_user_feat, batch_item_feat)
                        # 双损失计算
                        loss_distill = self.mse_criterion(pred, batch_teacher_score)
                        loss_label = self.bce_criterion(pred, batch_label)
                        loss = self.distill_weight * loss_distill + self.label_weight * loss_label
                    # 反向传播
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    pred = self.model(batch_user_feat, batch_item_feat)
                    loss_distill = self.mse_criterion(pred, batch_teacher_score)
                    loss_label = self.bce_criterion(pred, batch_label)
                    loss = self.distill_weight * loss_distill + self.label_weight * loss_label
                    loss.backward()
                    self.optimizer.step()

                # 损失统计
                total_loss += loss.item()
                total_distill_loss += loss_distill.item()
                total_label_loss += loss_label.item()

            # Epoch日志
            avg_loss = total_loss / len(train_data_loader)
            avg_distill_loss = total_distill_loss / len(train_data_loader)
            avg_label_loss = total_label_loss / len(train_data_loader)
            print(f"【蒸馏双塔】Batch训练完成，总Loss: {avg_loss:.6f}，蒸馏Loss: {avg_distill_loss:.6f}，标签Loss: {avg_label_loss:.6f}")

            # 验证集评估（仅当传入val_data且是最后一个epoch时）
            if val_data is not None and epoch == epochs-1:
                val_user_feat_np, val_item_feat_np, val_label_np = val_data
                val_user_tensor = torch.FloatTensor(val_user_feat_np).to(DEVICE)
                val_item_tensor = torch.FloatTensor(val_item_feat_np).to(DEVICE)
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(val_user_tensor, val_item_tensor).cpu().numpy().flatten()
                val_auc = roc_auc_score(val_label_np, val_pred)
                print(f"【蒸馏双塔】验证集 AUC: {val_auc:.6f}")
                self.model.train()

        self.is_fitted = True

    def predict(self, user_feat_np, item_feat_np):
        """批量打分，兼容numpy数组和torch张量，自动匹配返回类型，避免重复CPU-GPU拷贝"""
        self.model.eval()
        is_numpy_input = isinstance(user_feat_np, np.ndarray)
        
        # 统一处理输入，转为GPU张量计算
        if is_numpy_input:
            user_tensor = torch.FloatTensor(user_feat_np).to(DEVICE, non_blocking=True)
            item_tensor = torch.FloatTensor(item_feat_np).to(DEVICE, non_blocking=True)
        else:
            user_tensor = user_feat_np
            item_tensor = item_feat_np
        
        with torch.no_grad():
            pred = self.model(user_tensor, item_tensor).flatten()
        
        # 输入是numpy则返回numpy，输入是张量则返回张量，保证兼容性
        if is_numpy_input:
            pred = pred.detach().cpu().numpy()
        
        return pred
    
    def save(self):
        """保存蒸馏双塔模型与归一化器"""
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "distill_tower_model.pth"))
        with open(os.path.join(self.save_dir, "user_scaler.pkl"), "wb") as f:
            pickle.dump(self.user_scaler, f)
        with open(os.path.join(self.save_dir, "item_scaler.pkl"), "wb") as f:
            pickle.dump(self.item_scaler, f)
        # 保存权重配置
        with open(os.path.join(self.save_dir, "weight_config.pkl"), "wb") as f:
            pickle.dump({"distill_weight": self.distill_weight, "label_weight": self.label_weight}, f)
        print(f"【蒸馏双塔】模型保存完成！")

    @classmethod
    def load(cls):
        """加载已训练的蒸馏双塔模型"""
        model = cls()
        # 加载模型权重
        model.model.load_state_dict(torch.load(os.path.join(model.save_dir, "distill_tower_model.pth"), map_location=DEVICE))
        model.model.to(DEVICE)
        model.model.eval()
        # 加载归一化器
        with open(os.path.join(model.save_dir, "user_scaler.pkl"), "rb") as f:
            model.user_scaler = pickle.load(f)
        with open(os.path.join(model.save_dir, "item_scaler.pkl"), "rb") as f:
            model.item_scaler = pickle.load(f)
        # 加载权重配置
        with open(os.path.join(model.save_dir, "weight_config.pkl"), "rb") as f:
            weight_config = pickle.load(f)
            model.distill_weight = weight_config["distill_weight"]
            model.label_weight = weight_config["label_weight"]
        model.is_fitted = True
        print(f"【蒸馏双塔】模型加载完成！")
        return model

#  模型工厂（一键切换模型，方便扩展）
ROUGH_MODEL_MAP = {
    "FM": FMModel,
    "DistillDualTower": DistillDualTowerRoughRank
}

def get_rough_model(model_name):
    """模型获取工厂函数，新增模型只需在ROUGH_MODEL_MAP里注册即可"""
    if model_name not in ROUGH_MODEL_MAP:
        raise ValueError(f"模型{model_name}不存在，支持的模型：{list(ROUGH_MODEL_MAP.keys())}")
    return ROUGH_MODEL_MAP[model_name]()
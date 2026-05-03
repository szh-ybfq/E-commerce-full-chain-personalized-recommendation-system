import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import pickle
import gc
import warnings
warnings.filterwarnings('ignore')

#  全局配置（与召回/粗排完全对齐，保证链路一致性）
# 硬件配置
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
if USE_CUDA:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

# 训练超参
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
EMBEDDING_DIM = 64
EPOCHS = 3
TOP_N_PRECISION = 10  # 精排最终输出Top10商品
MAX_SEQ_LEN = 50       # DIN用户行为序列最大长度
TIME_DECAY_ALPHA = 0.8 # DIN时间衰减系数，符合文档优化要求

# 内存优化数据类型
DTYPE_CONFIG = {
    "user_id": np.int32,
    "item_id": np.int32,
    "category_id": np.int32,
    "float_feature": np.float32,
    "idx_dtype": np.int32
}

# 模型保存路径
BASE_SAVE_DIR = "./Saved/4_Precision"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

#  特征体系定义（与特征工程/粗排完全对齐）
# 用户侧连续特征
USER_CONTINUOUS_FEATS = [
    "user_total_click", "user_total_buy", "user_total_cart", "user_total_fav",
    "user_cvr", "user_ctr", "user_cart_rate", "user_fav_rate",
    "user_behavior_days", "user_is_new"
]
# 物品侧连续特征
ITEM_CONTINUOUS_FEATS = [
    "item_total_click", "item_total_buy", "item_total_cart", "item_total_fav",
    "item_cvr", "item_ctr", "item_cart_rate", "item_fav_rate",
    "item_is_hot", "item_is_new"
]
# 全量连续特征
CONTINUOUS_FEATS = USER_CONTINUOUS_FEATS + ITEM_CONTINUOUS_FEATS
# 离散特征，需Embedding编码
CATEGORICAL_FEATS = ["user_id", "item_id", "category_id"]
# 多任务标签定义：CTR/CVR/加购率三目标
MULTI_TASK_LABELS = ["click_label", "buy_label", "cart_label"]
LABEL_COL = "click_label" # 单任务默认标签

#  精排模型基类
class BasePrecisionRank(nn.Module):
    def __init__(self, top_n=TOP_N_PRECISION):
        super().__init__()
        self.top_n = top_n
        self.name = self.__class__.__name__
        self.save_dir = os.path.join(BASE_SAVE_DIR, self.name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 特征预处理组件
        self.continuous_scaler = StandardScaler() # 连续特征归一化
        self.cate_encoders = {col: LabelEncoder() for col in CATEGORICAL_FEATS} # 离散特征编码
        self.is_fitted = False
        
        # 推理加速缓存（预计算全量物品特征，消除重复计算）
        self.item_continuous_cache = None
        self.item_cate_cache = None
        self.item_id_to_idx = None

    #  核心抽象方法（子类必须实现）
    def fit(self, train_data_loader, val_data=None, epochs=1):
        """模型训练接口"""
        raise NotImplementedError

    def predict(self, continuous_feat_np, categorical_feat_np, seq_feat_np=None, seq_mask_np=None, seq_time_decay=None):
        """模型预测打分接口"""
        raise NotImplementedError

    def save(self):
        """模型与预处理组件保存"""
        raise NotImplementedError

    @classmethod
    def load(cls):
        """模型与预处理组件加载"""
        raise NotImplementedError

    #  单用户精排接口（兼容测试/调试，已修复特征维度不匹配问题）
    def rank(self, user_id, user_feature, rough_rank_item_list, item_features_df):
        """
        单用户精排：从粗排Top50中输出精排TopN
        :param user_id: 用户ID
        :param user_feature: 用户特征Series
        :param rough_rank_item_list: 粗排输出的Top50商品列表
        :param item_features_df: 全量商品特征DataFrame
        :return: 精排TopN商品ID列表
        """
        # 过滤有效商品
        valid_items = [item for item in rough_rank_item_list if item in item_features_df["item_id"].values]
        if len(valid_items) == 0:
            print(f"用户{user_id}无有效候选商品，返回空列表")
            return []
        
        # 1. 构建用户-商品对的全量特征，和训练逻辑完全对齐
        valid_item_df = item_features_df[item_features_df["item_id"].isin(valid_items)].reset_index(drop=True)
        # 重复用户特征，和商品数量对齐
        user_df = pd.DataFrame([user_feature] * len(valid_item_df)).reset_index(drop=True)
        # 拼接用户+商品特征，和训练时的特征顺序完全一致
        merge_df = pd.concat([user_df, valid_item_df], axis=1)
        # 去重列名，避免重复
        merge_df = merge_df.loc[:, ~merge_df.columns.duplicated()]
        
        # 2. 连续特征处理：和训练完全一致，传入20维全量连续特征做transform
        continuous_feat_np = self.continuous_scaler.transform(
            merge_df[CONTINUOUS_FEATS].values
        ).astype(DTYPE_CONFIG["float_feature"])
        
        # 3. 离散特征处理：和训练完全一致，按顺序编码，兜底unseen label
        categorical_feat_np = []
        for col in CATEGORICAL_FEATS:
            if col in merge_df.columns:
                encoder = self.cate_encoders[col]
                # 只保留训练过的标签，没见过的设为0
                valid_mask = merge_df[col].isin(encoder.classes_)
                encoded_col = np.zeros(len(merge_df), dtype=DTYPE_CONFIG["idx_dtype"])
                encoded_col[valid_mask] = encoder.transform(merge_df[col][valid_mask].values)
                categorical_feat_np.append(encoded_col)
        categorical_feat_np = np.stack(categorical_feat_np, axis=1).astype(DTYPE_CONFIG["idx_dtype"])
        
        # 4. 模型打分，MMoE模型只返回CTR打分用于排序
        scores = self.predict(continuous_feat_np, categorical_feat_np, return_ctr_only=True)
        
        # 5. 排序取TopN
        item_score_df = pd.DataFrame({"item_id": valid_item_df["item_id"].values, "score": scores})
        top_items = item_score_df.sort_values("score", ascending=False).head(self.top_n)["item_id"].tolist()
        return top_items
    
    #  推理加速：预构建全量物品特征缓存（全流程仅执行1次，修复编码兜底）
    def build_item_feature_cache(self, item_features_df):
        """
        预计算全量物品的原始特征与编码，建立ID到索引的映射，消除重复计算
        :param item_features_df: 全量商品特征DataFrame
        """
        print(f"===== 预构建物品特征缓存 =====")
        # 排序保证索引稳定
        item_features_df = item_features_df.sort_values("item_id").reset_index(drop=True)
        # 建立物品ID→索引的O(1)映射字典
        self.item_id_to_idx = pd.Series(item_features_df.index, index=item_features_df["item_id"]).to_dict()
        
        # 缓存物品原始连续特征（不提前做归一化，避免维度不匹配）
        self.item_continuous_cache = torch.FloatTensor(
            item_features_df[ITEM_CONTINUOUS_FEATS].values.astype(DTYPE_CONFIG["float_feature"])
        ).to(DEVICE)
        
        # 预计算离散特征（仅item_id、category_id，严格对齐训练编码逻辑）
        item_cate_feat = []
        for col in ["item_id", "category_id"]:
            encoder = self.cate_encoders[col]
            valid_mask = item_features_df[col].isin(encoder.classes_)
            encoded_col = np.zeros(len(item_features_df), dtype=DTYPE_CONFIG["idx_dtype"])
            encoded_col[valid_mask] = encoder.transform(item_features_df[col][valid_mask].values)
            item_cate_feat.append(encoded_col)
        self.item_cate_cache = torch.IntTensor(np.stack(item_cate_feat, axis=1)).to(DEVICE)
        
        print(f"物品特征缓存构建完成，共缓存 {len(self.item_continuous_cache)} 个商品特征")
        gc.collect()
    
    #  批量用户精排核心（GPU向量化加速，已修复特征维度不匹配问题）
    #  批量用户精排核心（GPU向量化加速，修复索引越界核心问题）
    def batch_rank(self, user_ids, user_features_df, rough_rank_item_lists, item_features_df=None):
        """
        全量用户批量精排，GPU并行加速，工业级线上推理标准方案
        :param user_ids: 批量用户ID列表
        :param user_features_df: 对应用户的特征DataFrame
        :param rough_rank_item_lists: 对应用户的粗排Top50商品列表（二维列表）
        :param item_features_df: 全量商品特征DataFrame（兼容未提前构建缓存的场景）
        :return: 每个用户的精排TopN商品ID列表
        """
        # 预构建物品特征缓存（如果没构建的话）
        if self.item_continuous_cache is None or self.item_id_to_idx is None:
            if item_features_df is None:
                raise ValueError("未预构建物品缓存时，必须传入item_features_df")
            self.build_item_feature_cache(item_features_df)
        
        # 1. 批量用户特征预处理 + user_id编码（核心修复点）
        user_features_df = user_features_df.reset_index(drop=True)
        # 对user_id做训练对齐的编码，unseen label兜底为0
        user_encoder = self.cate_encoders["user_id"]
        valid_user_mask = user_features_df["user_id"].isin(user_encoder.classes_)
        encoded_user_id = np.zeros(len(user_features_df), dtype=DTYPE_CONFIG["idx_dtype"])
        encoded_user_id[valid_user_mask] = user_encoder.transform(user_features_df["user_id"][valid_user_mask].values)
        user_features_df["encoded_user_id"] = encoded_user_id
        
        # 2. 展开用户-商品对，向量化处理
        all_user_idx = []
        all_item_idx = []
        user_item_count = []
        for user_idx, (uid, recall_items) in enumerate(zip(user_ids, rough_rank_item_lists)):
            # 过滤无效商品，O(1)索引查询
            valid_item_idx = [self.item_id_to_idx[item] for item in recall_items if item in self.item_id_to_idx]
            item_count = len(valid_item_idx)
            if item_count == 0:
                user_item_count.append(0)
                continue
            # 记录索引用于后续分组
            all_user_idx.extend([user_idx] * item_count)
            all_item_idx.extend(valid_item_idx)
            user_item_count.append(item_count)
        if len(all_user_idx) == 0:
            return [[] for _ in user_ids]
        
        # 3. 拼接用户+物品全量连续特征，和训练逻辑完全对齐（20维）
        batch_user_continuous = user_features_df[USER_CONTINUOUS_FEATS].values[all_user_idx]
        batch_item_continuous = self.item_continuous_cache[all_item_idx].cpu().numpy()
        batch_full_continuous = np.concatenate([batch_user_continuous, batch_item_continuous], axis=1)
        batch_continuous = torch.FloatTensor(
            self.continuous_scaler.transform(batch_full_continuous).astype(DTYPE_CONFIG["float_feature"])
        ).to(DEVICE)
        
        # 4. 构建全量离散特征（严格对齐训练顺序：user_id编码、item_id编码、category_id编码）
        batch_user_cate = torch.IntTensor(
            user_features_df[["encoded_user_id"]].values[all_user_idx]
        ).to(DEVICE)
        batch_item_cate = self.item_cate_cache[all_item_idx]
        batch_categorical = torch.concat([batch_user_cate, batch_item_cate], axis=1)
        
        # 5. GPU批量预测打分
        batch_scores = self.predict(batch_continuous, batch_categorical, return_ctr_only=True)
        
        # 6. 分组取TopN
        top_items_list = []
        ptr = 0
        item_id_list = list(self.item_id_to_idx.keys())
        for user_idx, item_count in enumerate(user_item_count):
            if item_count == 0:
                top_items_list.append([])
                continue
            user_scores = batch_scores[ptr:ptr+item_count]
            current_item_idx = all_item_idx[ptr:ptr+item_count]
            user_item_ids = [item_id_list[idx] for idx in current_item_idx]
            ptr += item_count
            top_k = min(self.top_n, item_count)
            top_idx = torch.argsort(user_scores, descending=True)[:top_k].cpu().numpy()
            top_items = [user_item_ids[i] for i in top_idx]
            top_items_list.append(top_items)
        
        # 释放显存/内存
        del batch_continuous, batch_categorical, batch_scores
        torch.cuda.empty_cache()
        gc.collect()
        return top_items_list
#  模型1：LR逻辑回归
class LRModel(BasePrecisionRank):
    def __init__(self, top_n=TOP_N_PRECISION):
        super().__init__(top_n)
        self.input_dim = len(CONTINUOUS_FEATS)
        # LR核心线性层
        self.model = nn.Linear(self.input_dim, 1, bias=True).to(DEVICE)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        self.criterion = nn.BCEWithLogitsLoss()
        # 混合精度训练
        if USE_CUDA:
            self.scaler = torch.cuda.amp.GradScaler()

    def fit(self, train_data_loader, val_data=None, epochs=1):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for batch_idx, (batch_continuous, batch_cate, batch_label) in enumerate(
                tqdm(train_data_loader, desc=f"【LR】训练 Batch", mininterval=1, leave=False)
            ):
                batch_continuous = batch_continuous.to(DEVICE, non_blocking=True)
                batch_label = batch_label.to(DEVICE, non_blocking=True).float().unsqueeze(1)
                self.optimizer.zero_grad(set_to_none=True)

                # 混合精度前向传播
                if USE_CUDA:
                    with torch.cuda.amp.autocast():
                        output = self.model(batch_continuous)
                        loss = self.criterion(output, batch_label)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    output = self.model(batch_continuous)
                    loss = self.criterion(output, batch_label)
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item()

            # 日志输出
            avg_loss = total_loss / len(train_data_loader)
            print(f"【LR】Epoch {epoch+1} 训练完成，平均Loss: {avg_loss:.6f}")
            # 验证集评估
            if val_data is not None and epoch == epochs-1:
                val_continuous, val_cate, val_label = val_data
                val_pred = self.predict(val_continuous, val_cate)
                val_auc = roc_auc_score(val_label, val_pred)
                print(f"【LR】验证集 AUC: {val_auc:.6f}")
                self.model.train()
        self.is_fitted = True

    def predict(self, continuous_feat_np, categorical_feat_np=None, seq_feat_np=None, seq_mask_np=None, seq_time_decay=None, return_ctr_only=False):
        self.model.eval()
        is_numpy_input = isinstance(continuous_feat_np, np.ndarray)
        # 转GPU张量
        if is_numpy_input:
            feat_tensor = torch.FloatTensor(continuous_feat_np).to(DEVICE, non_blocking=True)
        else:
            feat_tensor = continuous_feat_np
        
        with torch.no_grad():
            output = self.model(feat_tensor).flatten()
            pred = torch.sigmoid(output)
        
        if is_numpy_input:
            pred = pred.detach().cpu().numpy()
        return pred

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "lr_model.pth"))
        # 保存特征预处理组件
        with open(os.path.join(self.save_dir, "continuous_scaler.pkl"), "wb") as f:
            pickle.dump(self.continuous_scaler, f)
        with open(os.path.join(self.save_dir, "cate_encoders.pkl"), "wb") as f:
            pickle.dump(self.cate_encoders, f)
        print(f"【LR】模型保存完成！")

    @classmethod
    def load(cls):
        model = cls()
        # 加载模型权重
        model.model.load_state_dict(torch.load(os.path.join(model.save_dir, "lr_model.pth"), map_location=DEVICE))
        model.model.to(DEVICE)
        model.model.eval()
        # 加载特征预处理组件
        with open(os.path.join(model.save_dir, "continuous_scaler.pkl"), "rb") as f:
            model.continuous_scaler = pickle.load(f)
        with open(os.path.join(model.save_dir, "cate_encoders.pkl"), "rb") as f:
            model.cate_encoders = pickle.load(f)
        model.is_fitted = True
        print(f"【LR】模型加载完成！")
        return model

#  模型2：DeepFM
class DeepFMModel(BasePrecisionRank):
    def __init__(self, top_n=TOP_N_PRECISION):
        super().__init__(top_n)
        self.continuous_dim = len(CONTINUOUS_FEATS)
        self.cate_dim = len(CATEGORICAL_FEATS)
        self.total_feature_dim = self.continuous_dim + self.cate_dim
        self.hidden_dims = [256, 128, 64]
        # 离散特征Embedding层（延迟初始化）
        self.embedding_layers = nn.ModuleDict()
        self._is_embedding_init = False

        # FM线性层
        self.fm_linear = nn.Linear(self.total_feature_dim, 1)
        # Deep侧DNN
        deep_input_dim = self.continuous_dim + EMBEDDING_DIM * self.cate_dim
        deep_layers = []
        input_dim = deep_input_dim
        for hidden_dim in self.hidden_dims:
            deep_layers.append(nn.Linear(input_dim, hidden_dim))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.LayerNorm(hidden_dim))
            deep_layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        deep_layers.append(nn.Linear(input_dim, 1))
        self.deep_dnn = nn.Sequential(*deep_layers)

        self.model = nn.ModuleList([self.embedding_layers, self.fm_linear, self.deep_dnn]).to(DEVICE)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        self.criterion = nn.BCEWithLogitsLoss()
        if USE_CUDA:
            self.scaler = torch.cuda.amp.GradScaler()

    def _init_embedding_layers(self):
        """延迟初始化Embedding层，适配LabelEncoder的vocab_size"""
        for col in CATEGORICAL_FEATS:
            vocab_size = len(self.cate_encoders[col].classes_)
            self.embedding_layers[col] = nn.Embedding(vocab_size, EMBEDDING_DIM).to(DEVICE)
        self._is_embedding_init = True
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    def _fm_forward(self, linear_input, embed_x):
        # FM线性项
        linear_part = self.fm_linear(linear_input)
        # FM二阶交叉项
        square_of_sum = torch.sum(embed_x, dim=1) ** 2
        sum_of_square = torch.sum(embed_x ** 2, dim=1)
        cross_part = 0.5 * (square_of_sum - sum_of_square).sum(dim=1, keepdim=True)
        return linear_part + cross_part

    def fit(self, train_data_loader, val_data=None, epochs=1):
        if not self._is_embedding_init:
            self._init_embedding_layers()
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for batch_idx, (batch_continuous, batch_cate, batch_label) in enumerate(
                tqdm(train_data_loader, desc=f"【DeepFM】训练 Batch", mininterval=1, leave=False)
            ):
                batch_continuous = batch_continuous.to(DEVICE, non_blocking=True)
                batch_cate = batch_cate.to(DEVICE, non_blocking=True).long()
                batch_label = batch_label.to(DEVICE, non_blocking=True).float().unsqueeze(1)
                self.optimizer.zero_grad(set_to_none=True)

                # 特征拼接
                linear_input = torch.concat([batch_continuous, batch_cate.float()], axis=1)
                # 离散特征Embedding
                embed_list = []
                for i, col in enumerate(CATEGORICAL_FEATS):
                    embed = self.embedding_layers[col](batch_cate[:, i])
                    embed_list.append(embed.unsqueeze(1))
                embed_concat = torch.concat(embed_list, axis=1)
                # FM前向
                fm_out = self._fm_forward(linear_input, embed_concat)
                # Deep前向
                deep_embed = embed_concat.flatten(start_dim=1)
                deep_input = torch.concat([batch_continuous, deep_embed], axis=1)
                deep_out = self.deep_dnn(deep_input)
                # 合并输出
                final_out = fm_out + deep_out

                # 混合精度反向传播
                if USE_CUDA:
                    with torch.cuda.amp.autocast():
                        loss = self.criterion(final_out, batch_label)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss = self.criterion(final_out, batch_label)
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_data_loader)
            print(f"【DeepFM】Epoch {epoch+1} 训练完成，平均Loss: {avg_loss:.6f}")
            if val_data is not None and epoch == epochs-1:
                val_continuous, val_cate, val_label = val_data
                val_pred = self.predict(val_continuous, val_cate)
                val_auc = roc_auc_score(val_label, val_pred)
                print(f"【DeepFM】验证集 AUC: {val_auc:.6f}")
                self.model.train()
        self.is_fitted = True

    def predict(self, continuous_feat_np, categorical_feat_np, seq_feat_np=None, seq_mask_np=None, seq_time_decay=None, return_ctr_only=False):
        self.model.eval()
        is_numpy_input = isinstance(continuous_feat_np, np.ndarray)
        if is_numpy_input:
            continuous_tensor = torch.FloatTensor(continuous_feat_np).to(DEVICE, non_blocking=True)
            cate_tensor = torch.IntTensor(categorical_feat_np).to(DEVICE, non_blocking=True).long()
        else:
            continuous_tensor = continuous_feat_np
            cate_tensor = categorical_feat_np.long()
        
        with torch.no_grad():
            linear_input = torch.concat([continuous_tensor, cate_tensor.float()], axis=1)
            embed_list = []
            for i, col in enumerate(CATEGORICAL_FEATS):
                embed = self.embedding_layers[col](cate_tensor[:, i])
                embed_list.append(embed.unsqueeze(1))
            embed_concat = torch.concat(embed_list, axis=1)
            fm_out = self._fm_forward(linear_input, embed_concat)
            deep_embed = embed_concat.flatten(start_dim=1)
            deep_input = torch.concat([continuous_tensor, deep_embed], axis=1)
            deep_out = self.deep_dnn(deep_input)
            final_out = fm_out + deep_out
            pred = torch.sigmoid(final_out.flatten())
        
        if is_numpy_input:
            pred = pred.detach().cpu().numpy()
        return pred

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "deepfm_model.pth"))
        with open(os.path.join(self.save_dir, "continuous_scaler.pkl"), "wb") as f:
            pickle.dump(self.continuous_scaler, f)
        with open(os.path.join(self.save_dir, "cate_encoders.pkl"), "wb") as f:
            pickle.dump(self.cate_encoders, f)
        print(f"【DeepFM】模型保存完成！")

    @classmethod
    def load(cls):
        model = cls()
        with open(os.path.join(model.save_dir, "continuous_scaler.pkl"), "rb") as f:
            model.continuous_scaler = pickle.load(f)
        with open(os.path.join(model.save_dir, "cate_encoders.pkl"), "rb") as f:
            model.cate_encoders = pickle.load(f)
        model._init_embedding_layers()
        model.model.load_state_dict(torch.load(os.path.join(model.save_dir, "deepfm_model.pth"), map_location=DEVICE))
        model.model.to(DEVICE)
        model.model.eval()
        model.is_fitted = True
        print(f"【DeepFM】模型加载完成！")
        return model

#  模型3：DIN
class DINAttentionLayer(nn.Module):
    """DIN注意力层，严格对齐文档：加入时间衰减优化，捕捉用户动态兴趣"""
    def __init__(self, embedding_dim, hidden_dims=[64, 32]):
        super().__init__()
        # 注意力MLP：输入=候选emb + 行为emb + 外积（符合文档要求）
        input_dim = embedding_dim * 3
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.attention_mlp = nn.Sequential(*layers)

    def forward(self, candidate_emb, seq_emb, seq_mask, seq_time_decay=None):
        """
        :param candidate_emb: 候选商品embedding [batch_size, embed_dim]
        :param seq_emb: 用户行为序列embedding [batch_size, max_seq_len, embed_dim]
        :param seq_mask: 序列padding掩码 [batch_size, max_seq_len]
        :param seq_time_decay: 时间衰减权重[batch_size, max_seq_len]
        :return: 注意力加权后的用户兴趣embedding [batch_size, embed_dim]
        """
        batch_size, max_seq_len, embed_dim = seq_emb.shape
        # 扩展候选商品embedding，和序列对齐
        candidate_expand = candidate_emb.unsqueeze(1).repeat(1, max_seq_len, 1)
        # 拼接特征：候选emb、序列emb、外积
        attention_input = torch.concat([
            candidate_expand, seq_emb, candidate_expand * seq_emb
        ], axis=-1)
        # 注意力打分
        attention_score = self.attention_mlp(attention_input).squeeze(-1)
        # 加入时间衰减：给近期行为更高权重
        if seq_time_decay is not None:
            attention_score = attention_score * seq_time_decay
        # padding掩码：无效位置打分置为负无穷
        attention_score = attention_score.masked_fill(seq_mask == 0, -1e9)
        attention_weight = torch.softmax(attention_score, dim=-1).unsqueeze(-1)
        # 加权求和得到用户兴趣embedding
        interest_emb = torch.sum(seq_emb * attention_weight, dim=1)
        return interest_emb

class DINModel(BasePrecisionRank):
    def __init__(self, top_n=TOP_N_PRECISION):
        super().__init__(top_n)
        self.max_seq_len = MAX_SEQ_LEN
        self.continuous_dim = len(CONTINUOUS_FEATS)
        self.cate_dim = len(CATEGORICAL_FEATS)
        self.hidden_dims = [256, 128, 64]
        # 离散特征Embedding层（延迟初始化）
        self.embedding_layers = nn.ModuleDict()
        self._is_embedding_init = False

        # DIN注意力层
        self.attention_layer = DINAttentionLayer(EMBEDDING_DIM)
        # 最终DNN打分层
        dnn_input_dim = self.continuous_dim + EMBEDDING_DIM * (self.cate_dim + 1)
        dnn_layers = []
        input_dim = dnn_input_dim
        for hidden_dim in self.hidden_dims:
            dnn_layers.append(nn.Linear(input_dim, hidden_dim))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.LayerNorm(hidden_dim))
            dnn_layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        dnn_layers.append(nn.Linear(input_dim, 1))
        self.dnn_scorer = nn.Sequential(*dnn_layers)

        self.model = nn.ModuleList([self.embedding_layers, self.attention_layer, self.dnn_scorer]).to(DEVICE)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        self.criterion = nn.BCEWithLogitsLoss()
        if USE_CUDA:
            self.scaler = torch.cuda.amp.GradScaler()

    def _init_embedding_layers(self):
        """延迟初始化Embedding层，适配LabelEncoder的vocab_size"""
        for col in CATEGORICAL_FEATS:
            vocab_size = len(self.cate_encoders[col].classes_)
            self.embedding_layers[col] = nn.Embedding(vocab_size, EMBEDDING_DIM).to(DEVICE)
        self._is_embedding_init = True
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    def fit(self, train_data_loader, val_data=None, epochs=1):
        if not self._is_embedding_init:
            self._init_embedding_layers()
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for batch_idx, (batch_continuous, batch_cate, batch_seq, batch_seq_mask, batch_seq_decay, batch_label) in enumerate(
                tqdm(train_data_loader, desc=f"【DIN】训练 Batch", mininterval=1, leave=False)
            ):
                batch_continuous = batch_continuous.to(DEVICE, non_blocking=True)
                batch_cate = batch_cate.to(DEVICE, non_blocking=True).long()
                batch_seq = batch_seq.to(DEVICE, non_blocking=True).long()
                batch_seq_mask = batch_seq_mask.to(DEVICE, non_blocking=True)
                batch_seq_decay = batch_seq_decay.to(DEVICE, non_blocking=True)
                batch_label = batch_label.to(DEVICE, non_blocking=True).float().unsqueeze(1)
                self.optimizer.zero_grad(set_to_none=True)

                # 1. 离散特征Embedding
                embed_list = []
                for i, col in enumerate(CATEGORICAL_FEATS):
                    embed = self.embedding_layers[col](batch_cate[:, i])
                    embed_list.append(embed)
                base_embed_concat = torch.concat(embed_list, axis=1)
                # 2. 候选商品Embedding
                item_emb = self.embedding_layers["item_id"](batch_cate[:, CATEGORICAL_FEATS.index("item_id")])
                # 3. 行为序列Embedding + 注意力加权（带时间衰减）
                seq_emb = self.embedding_layers["item_id"](batch_seq)
                interest_emb = self.attention_layer(item_emb, seq_emb, batch_seq_mask, batch_seq_decay)
                # 4. 全特征拼接
                dnn_input = torch.concat([batch_continuous, base_embed_concat, interest_emb], axis=1)
                # 5. 打分
                final_out = self.dnn_scorer(dnn_input)

                # 混合精度反向传播
                if USE_CUDA:
                    with torch.cuda.amp.autocast():
                        loss = self.criterion(final_out, batch_label)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss = self.criterion(final_out, batch_label)
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_data_loader)
            print(f"【DIN】Epoch {epoch+1} 训练完成，平均Loss: {avg_loss:.6f}")
            if val_data is not None and epoch == epochs-1:
                val_continuous, val_cate, val_seq, val_seq_mask, val_seq_decay, val_label = val_data
                val_pred = self.predict(val_continuous, val_cate, val_seq, val_seq_mask, val_seq_decay)
                val_auc = roc_auc_score(val_label, val_pred)
                print(f"【DIN】验证集 AUC: {val_auc:.6f}")
                self.model.train()
        self.is_fitted = True

    def predict(self, continuous_feat_np, categorical_feat_np, seq_feat_np=None, seq_mask_np=None, seq_time_decay=None, return_ctr_only=False):
        self.model.eval()
        is_numpy_input = isinstance(continuous_feat_np, np.ndarray)
        if is_numpy_input:
            continuous_tensor = torch.FloatTensor(continuous_feat_np).to(DEVICE, non_blocking=True)
            cate_tensor = torch.IntTensor(categorical_feat_np).to(DEVICE, non_blocking=True).long()
            seq_tensor = torch.IntTensor(seq_feat_np).to(DEVICE, non_blocking=True).long() if seq_feat_np is not None else None
            seq_mask_tensor = torch.FloatTensor(seq_mask_np).to(DEVICE, non_blocking=True) if seq_mask_np is not None else None
            seq_decay_tensor = torch.FloatTensor(seq_time_decay).to(DEVICE, non_blocking=True) if seq_time_decay is not None else None
        else:
            continuous_tensor = continuous_feat_np
            cate_tensor = categorical_feat_np.long()
            seq_tensor = seq_feat_np.long() if seq_feat_np is not None else None
            seq_mask_tensor = seq_mask_np
            seq_decay_tensor = seq_time_decay
        
        with torch.no_grad():
            # 基础特征Embedding
            embed_list = []
            for i, col in enumerate(CATEGORICAL_FEATS):
                embed = self.embedding_layers[col](cate_tensor[:, i])
                embed_list.append(embed)
            base_embed_concat = torch.concat(embed_list, axis=1)
            # 候选商品Embedding
            item_emb = self.embedding_layers["item_id"](cate_tensor[:, CATEGORICAL_FEATS.index("item_id")])
            # 注意力加权兴趣Embedding
            seq_emb = self.embedding_layers["item_id"](seq_tensor)
            interest_emb = self.attention_layer(item_emb, seq_emb, seq_mask_tensor, seq_decay_tensor)
            # 全特征拼接与打分
            dnn_input = torch.concat([continuous_tensor, base_embed_concat, interest_emb], axis=1)
            final_out = self.dnn_scorer(dnn_input)
            pred = torch.sigmoid(final_out.flatten())
        
        if is_numpy_input:
            pred = pred.detach().cpu().numpy()
        return pred

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "din_model.pth"))
        with open(os.path.join(self.save_dir, "continuous_scaler.pkl"), "wb") as f:
            pickle.dump(self.continuous_scaler, f)
        with open(os.path.join(self.save_dir, "cate_encoders.pkl"), "wb") as f:
            pickle.dump(self.cate_encoders, f)
        print(f"【DIN】模型保存完成！")

    @classmethod
    def load(cls):
        model = cls()
        with open(os.path.join(model.save_dir, "continuous_scaler.pkl"), "rb") as f:
            model.continuous_scaler = pickle.load(f)
        with open(os.path.join(model.save_dir, "cate_encoders.pkl"), "rb") as f:
            model.cate_encoders = pickle.load(f)
        model._init_embedding_layers()
        model.model.load_state_dict(torch.load(os.path.join(model.save_dir, "din_model.pth"), map_location=DEVICE))
        model.model.to(DEVICE)
        model.model.eval()
        model.is_fitted = True
        print(f"【DIN】模型加载完成！")
        return model

#  模型4：MMoE
class ExpertLayer(nn.Module):
    """MMoE专家层：底层共享专家网络，符合文档8专家配置"""
    def __init__(self, input_dim, expert_num=8, hidden_dim=128):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(expert_num)
        ])

    def forward(self, x):
        """
        :param x: 输入特征 [batch_size, input_dim]
        :return: 所有专家的输出 [batch_size, expert_num, hidden_dim]
        """
        expert_out = [expert(x) for expert in self.experts]
        return torch.stack(expert_out, dim=1)

class GateLayer(nn.Module):
    """MMoE门控层：每个任务独立门控，自动选择专家"""
    def __init__(self, input_dim, expert_num=8):
        super().__init__()
        self.gate = nn.Linear(input_dim, expert_num)

    def forward(self, x, expert_out):
        """
        :param x: 输入特征 [batch_size, input_dim]
        :param expert_out: 专家层输出 [batch_size, expert_num, hidden_dim]
        :return: 门控加权后的专家输出 [batch_size, hidden_dim]
        """
        gate_weight = torch.softmax(self.gate(x), dim=-1).unsqueeze(-1)
        weighted_out = torch.sum(expert_out * gate_weight, dim=1)
        return weighted_out

class TaskTower(nn.Module):
    """MMoE任务塔：每个任务独立预测塔，对应CTR/CVR/加购率三目标"""
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super().__init__()
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.tower = nn.Sequential(*layers)

    def forward(self, x):
        return self.tower(x)

class MMoEModel(BasePrecisionRank):
    def __init__(self, top_n=TOP_N_PRECISION, expert_num=8, task_names=MULTI_TASK_LABELS):
        super().__init__(top_n)
        self.task_names = task_names
        self.task_num = len(task_names)
        self.expert_num = expert_num
        self.expert_hidden_dim = 128
        self.tower_hidden_dims = [64, 32]
        # 离散特征Embedding层（延迟初始化）
        self.embedding_layers = nn.ModuleDict()
        self._is_embedding_init = False

        # 输入维度：连续特征 + 离散特征Embedding展平
        self.input_dim = len(CONTINUOUS_FEATS) + EMBEDDING_DIM * len(CATEGORICAL_FEATS)
        # MMoE核心层
        self.expert_layer = ExpertLayer(self.input_dim, expert_num, self.expert_hidden_dim)
        self.gate_layers = nn.ModuleList([GateLayer(self.input_dim, expert_num) for _ in range(self.task_num)])
        self.task_towers = nn.ModuleList([TaskTower(self.expert_hidden_dim, self.tower_hidden_dims) for _ in range(self.task_num)])

        # 不确定度加权：文档核心优化点，自动平衡多任务Loss，解决跷跷板效应
        self.task_log_vars = nn.Parameter(torch.zeros(self.task_num))
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

        # 整体模型移到GPU
        self.model = nn.ModuleList([
            self.embedding_layers, self.expert_layer, self.gate_layers, self.task_towers
        ]).to(DEVICE)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        if USE_CUDA:
            self.scaler = torch.cuda.amp.GradScaler()

    def _init_embedding_layers(self):
        """延迟初始化Embedding层，适配LabelEncoder的vocab_size"""
        for col in CATEGORICAL_FEATS:
            vocab_size = len(self.cate_encoders[col].classes_)
            self.embedding_layers[col] = nn.Embedding(vocab_size, EMBEDDING_DIM).to(DEVICE)
        self._is_embedding_init = True
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    def _uncertainty_weighted_loss(self, task_losses):
        """
        不确定度加权Loss（文档核心优化点）
        :param task_losses: 每个任务的Loss [task_num]
        :return: 加权后的总Loss，每个任务的加权Loss
        """
        total_loss = 0.0
        weighted_losses = []
        for i in range(self.task_num):
            precision = torch.exp(-self.task_log_vars[i])
            loss = precision * task_losses[i] + self.task_log_vars[i]
            total_loss += loss
            weighted_losses.append(loss)
        return total_loss, weighted_losses

    def fit(self, train_data_loader, val_data=None, epochs=1):
        if not self._is_embedding_init:
            self._init_embedding_layers()
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            task_total_losses = [0.0 for _ in range(self.task_num)]
            for batch_idx, (batch_continuous, batch_cate, batch_multi_labels) in enumerate(
                tqdm(train_data_loader, desc=f"【MMoE】训练 Batch", mininterval=1, leave=False)
            ):
                batch_continuous = batch_continuous.to(DEVICE, non_blocking=True)
                batch_cate = batch_cate.to(DEVICE, non_blocking=True).long()
                batch_multi_labels = batch_multi_labels.to(DEVICE, non_blocking=True).float()
                self.optimizer.zero_grad(set_to_none=True)

                # 1. 特征Embedding与拼接（向量化）
                embed_list = []
                for i, col in enumerate(CATEGORICAL_FEATS):
                    embed = self.embedding_layers[col](batch_cate[:, i])
                    embed_list.append(embed)
                embed_concat = torch.concat(embed_list, axis=1)
                model_input = torch.concat([batch_continuous, embed_concat], axis=1)

                # 2. MMoE前向
                expert_out = self.expert_layer(model_input)
                task_outputs = []
                for i in range(self.task_num):
                    gate_out = self.gate_layers[i](model_input, expert_out)
                    task_out = self.task_towers[i](gate_out)
                    task_outputs.append(task_out)

                # 3. 每个任务Loss计算 + 不确定度加权
                task_losses = []
                for i in range(self.task_num):
                    task_label = batch_multi_labels[:, i].unsqueeze(1)
                    # CVR任务仅用点击过的样本计算Loss，避免样本选择偏差
                    if self.task_names[i] == "buy_label":
                        click_mask = batch_multi_labels[:, 0] == 1  # click_label=1的样本
                        if torch.sum(click_mask) > 0:
                            task_loss = self.criterion(task_outputs[i][click_mask], task_label[click_mask]).mean()
                        else:
                            task_loss = torch.tensor(0.0, device=DEVICE)
                    else:
                        task_loss = self.criterion(task_outputs[i], task_label).mean()
                    task_losses.append(task_loss)
                    task_total_losses[i] += task_loss.item()

                # 4. 总Loss计算与反向传播
                total_batch_loss, weighted_batch_losses = self._uncertainty_weighted_loss(task_losses)
                if USE_CUDA:
                    with torch.cuda.amp.autocast():
                        loss = total_batch_loss
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss = total_batch_loss
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item()

            # Epoch日志
            avg_loss = total_loss / len(train_data_loader)
            avg_task_losses = [tl / len(train_data_loader) for tl in task_total_losses]
            print(f"【MMoE】Epoch {epoch+1} 训练完成，总Loss: {avg_loss:.6f}")
            for i, task_name in enumerate(self.task_names):
                print(f"  【{task_name}】平均Loss: {avg_task_losses[i]:.6f}")

            # 验证集评估
            if val_data is not None and epoch == epochs-1:
                val_continuous, val_cate, val_multi_labels = val_data
                val_preds = self.predict(val_continuous, val_cate)
                print(f"【MMoE】验证集评估结果:")
                for i, task_name in enumerate(self.task_names):
                    task_auc = roc_auc_score(val_multi_labels[:, i], val_preds[i])
                    print(f"  【{task_name}】AUC: {task_auc:.6f}")
                self.model.train()
        self.is_fitted = True

    def predict(self, continuous_feat_np, categorical_feat_np, seq_feat_np=None, seq_mask_np=None, seq_time_decay=None, return_ctr_only=False):
        """
        预测接口：单任务排序场景默认返回CTR任务打分，多任务返回所有任务结果
        :param return_ctr_only: 是否只返回CTR任务的打分（用于排序场景）
        """
        self.model.eval()
        is_numpy_input = isinstance(continuous_feat_np, np.ndarray)
        if is_numpy_input:
            continuous_tensor = torch.FloatTensor(continuous_feat_np).to(DEVICE, non_blocking=True)
            cate_tensor = torch.IntTensor(categorical_feat_np).to(DEVICE, non_blocking=True).long()
        else:
            continuous_tensor = continuous_feat_np
            cate_tensor = categorical_feat_np.long()
        
        with torch.no_grad():
            # 特征Embedding与拼接
            embed_list = []
            for i, col in enumerate(CATEGORICAL_FEATS):
                embed = self.embedding_layers[col](cate_tensor[:, i])
                embed_list.append(embed)
            embed_concat = torch.concat(embed_list, axis=1)
            model_input = torch.concat([continuous_tensor, embed_concat], axis=1)
            # MMoE前向
            expert_out = self.expert_layer(model_input)
            task_outputs = []
            for i in range(self.task_num):
                gate_out = self.gate_layers[i](model_input, expert_out)
                task_out = self.task_towers[i](gate_out)
                task_pred = torch.sigmoid(task_out.flatten())
                task_outputs.append(task_pred)
        # 单任务排序场景默认返回CTR打分，多任务返回所有结果
        if is_numpy_input:
            task_outputs = [pred.detach().cpu().numpy() for pred in task_outputs]
        
        # 排序场景只需要CTR打分，返回第一个元素
        if return_ctr_only:
            return task_outputs[0]
        # 评估场景返回所有任务的列表
        return task_outputs
    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "mmoe_model.pth"))
        with open(os.path.join(self.save_dir, "continuous_scaler.pkl"), "wb") as f:
            pickle.dump(self.continuous_scaler, f)
        with open(os.path.join(self.save_dir, "cate_encoders.pkl"), "wb") as f:
            pickle.dump(self.cate_encoders, f)
        # 保存任务配置
        with open(os.path.join(self.save_dir, "task_config.pkl"), "wb") as f:
            pickle.dump({"task_names": self.task_names, "expert_num": self.expert_num}, f)
        print(f"【MMoE】模型保存完成！")

    @classmethod
    def load(cls):
        model = cls()
        # 加载配置
        with open(os.path.join(model.save_dir, "task_config.pkl"), "rb") as f:
            task_config = pickle.load(f)
            model.task_names = task_config["task_names"]
            model.expert_num = task_config["expert_num"]
            model.task_num = len(model.task_names)
        # 加载预处理组件
        with open(os.path.join(model.save_dir, "continuous_scaler.pkl"), "rb") as f:
            model.continuous_scaler = pickle.load(f)
        with open(os.path.join(model.save_dir, "cate_encoders.pkl"), "rb") as f:
            model.cate_encoders = pickle.load(f)
        # 初始化Embedding层
        model._init_embedding_layers()
        # 加载模型权重
        model.model.load_state_dict(torch.load(os.path.join(model.save_dir, "mmoe_model.pth"), map_location=DEVICE))
        model.model.to(DEVICE)
        model.model.eval()
        model.is_fitted = True
        print(f"【MMoE】模型加载完成！")
        return model

#  模型工厂：一键切换模型，和粗排完全对齐 
PRECISION_MODEL_MAP = {
    "LR": LRModel,
    "DeepFM": DeepFMModel,
    "DIN": DINModel,
    "MMoE": MMoEModel
}

def get_precision_model(model_name):
    """
    模型获取工厂函数，新增模型只需在PRECISION_MODEL_MAP注册即可
    :param model_name: 模型名称，可选 LR/DeepFM/DIN/MMoE
    :return: 初始化后的精排模型
    """
    if model_name not in PRECISION_MODEL_MAP:
        raise ValueError(f"模型{model_name}不存在，支持的模型：{list(PRECISION_MODEL_MAP.keys())}")
    return PRECISION_MODEL_MAP[model_name]()
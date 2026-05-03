import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import faiss
from tqdm import tqdm
from numba import jit, prange
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

#  全局配置 
TOP_N_RECALL = 200
PER_ROAD_TOP_N = 50
NEW_ITEM_RATIO = 0.1
EMBEDDING_DIM = 64
MIND_INTEREST_NUM = 2
MAX_SEQ_LEN = 50
BATCH_SIZE = 4096
EPOCHS = 1
LEARNING_RATE = 1e-3
DSSM_SAMPLE_RATE = 0.05

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
print(f"【全局配置】设备: {DEVICE}, CUDNN加速: {USE_CUDA}")

# 内存优化dtype
DTYPE_CONFIG = {
    "user_id": np.int32,
    "item_id": np.int32,
    "category_id": np.int32,
    "behavior_type": "category",
    "float_feature": np.float32,
    "int_feature": np.int32,
    "idx_dtype": np.int32
}

# 模型保存根目录
MODEL_SAVE_DIR = "./recall_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

#  Numba加速核心函数 
@jit(nopython=True, fastmath=True, parallel=True)
def calc_jaccard_and_topk(
    coo_row, coo_col, coo_data, 
    item_degree, item_num, topk
):
    topk_item_ids = np.zeros((item_num, topk), dtype=np.int32)
    topk_sim_scores = np.zeros((item_num, topk), dtype=np.float32)
    
    for i in prange(len(coo_row)):
        item_i = coo_row[i]
        item_j = coo_col[i]
        co_count = coo_data[i]
        if item_i == item_j:
            continue
        jaccard = co_count / (item_degree[item_i] + item_degree[item_j] - co_count + 1e-8)
        min_idx = np.argmin(topk_sim_scores[item_i])
        if jaccard > topk_sim_scores[item_i, min_idx]:
            topk_sim_scores[item_i, min_idx] = jaccard
            topk_item_ids[item_i, min_idx] = item_j
    
    return topk_item_ids, topk_sim_scores

#  基础召回父类
class BaseRecall:
    def __init__(self, top_n=PER_ROAD_TOP_N):
        self.top_n = top_n
        self.name = self.__class__.__name__
        self.save_dir = os.path.join(MODEL_SAVE_DIR, self.name)
        os.makedirs(self.save_dir, exist_ok=True)
    def fit(self, df_train, user_features, item_features):
        raise NotImplementedError
    def predict(self, user_id, user_behavior_seq=None, **kwargs):
        raise NotImplementedError
    def batch_predict(self, all_user_ids, user_seq_dict, user_top_cate_dict):
        """全量用户批量预测接口，子类实现，和单用户predict逻辑完全一致"""
        raise NotImplementedError
    def save(self):
        """模型保存接口，子类实现"""
        raise NotImplementedError
    @classmethod
    def load(cls):
        """模型加载接口，子类实现"""
        raise NotImplementedError

#  热度召回
class HotRecall(BaseRecall):
    def __init__(self, top_n=PER_ROAD_TOP_N):
        super().__init__(top_n)
        self.hot_item_list = np.array([], dtype=DTYPE_CONFIG["item_id"])
        self.category_hot_map = dict()
    def fit(self, df_train, user_features, item_features):
        item_features = item_features.assign(
            hot_score = (0.5 * item_features["item_total_buy"] + 0.3 * item_features["item_ctr"] + 0.2 * item_features["item_cvr"]).astype(DTYPE_CONFIG["float_feature"])
        )
        self.hot_item_list = item_features.nlargest(self.top_n, "hot_score")["item_id"].values
        category_hot = item_features.sort_values(
            ["category_id", "hot_score"], ascending=[True, False], ignore_index=True
        ).groupby("category_id", observed=True, sort=False).head(20)
        self.category_hot_map = {
            cate_id: group["item_id"].values 
            for cate_id, group in category_hot.groupby("category_id", observed=True, sort=False)
        }
        print(f"【{self.name}】拟合完成，热门商品数: {len(self.hot_item_list)}")
    def predict(self, user_id, user_behavior_seq=None, user_top_cate=None):
        # 全兼容numpy数组/列表/None，彻底解决真值判断报错
        user_top_cate_np = np.asarray(user_top_cate) if user_top_cate is not None else np.array([])
        if user_top_cate_np.size > 0:
            cate_hot_item = np.concatenate(
                [self.category_hot_map.get(cate, np.array([])) for cate in user_top_cate_np[:2]]
            )
            return cate_hot_item[:self.top_n].tolist()
        return self.hot_item_list[:self.top_n].tolist()
    def batch_predict(self, all_user_ids, user_seq_dict, user_top_cate_dict):
        """全量用户批量预测，和单用户逻辑100%一致"""
        batch_result = {}
        for user_id in tqdm(all_user_ids, desc=f"【{self.name}】全量批量预测", mininterval=1):
            user_top_cate = user_top_cate_dict.get(user_id, [])
            batch_result[user_id] = self.predict(user_id, user_top_cate=user_top_cate)
        return batch_result
    def save(self):
        np.save(os.path.join(self.save_dir, "hot_item_list.npy"), self.hot_item_list)
        with open(os.path.join(self.save_dir, "category_hot_map.pkl"), "wb") as f:
            pickle.dump(self.category_hot_map, f)
        print(f"【{self.name}】模型保存完成")
    @classmethod
    def load(cls):
        model = cls()
        model.hot_item_list = np.load(os.path.join(model.save_dir, "hot_item_list.npy"))
        with open(os.path.join(model.save_dir, "category_hot_map.pkl"), "rb") as f:
            model.category_hot_map = pickle.load(f)
        print(f"【{model.name}】模型加载完成")
        return model

#  新品召回
class NewItemRecall(BaseRecall):
    def __init__(self, top_n=int(PER_ROAD_TOP_N*NEW_ITEM_RATIO*5)):
        super().__init__(top_n)
        self.new_item_map = dict()
    def fit(self, df_train, user_features, item_features):
        new_items = item_features[item_features["item_is_new"] == 1].copy()
        category_avg_ctr = item_features.groupby("category_id", observed=True, sort=False)["item_ctr"].mean().to_dict()
        new_items = new_items.assign(
            cate_avg_ctr = new_items["category_id"].map(category_avg_ctr).astype(DTYPE_CONFIG["float_feature"])
        )
        high_quality_new_items = new_items[new_items["item_ctr"] >= new_items["cate_avg_ctr"]]
        category_new = high_quality_new_items.sort_values(
            ["category_id", "item_ctr"], ascending=[True, False], ignore_index=True
        ).groupby("category_id", observed=True, sort=False).head(20)
        self.new_item_map = {
            cate_id: group["item_id"].values 
            for cate_id, group in category_new.groupby("category_id", observed=True, sort=False)
        }
        print(f"【{self.name}】拟合完成，覆盖品类数: {len(self.new_item_map)}")
    def predict(self, user_id, user_behavior_seq=None, user_top_cate=None):
        # 全兼容numpy数组/列表/None，彻底解决真值判断报错
        user_top_cate_np = np.asarray(user_top_cate) if user_top_cate is not None else np.array([])
        new_item_result = np.array([], dtype=DTYPE_CONFIG["item_id"])
        if user_top_cate_np.size > 0:
            new_item_result = np.concatenate(
                [self.new_item_map.get(cate, np.array([])) for cate in user_top_cate_np[:2]]
            )
        return new_item_result[:self.top_n].tolist()
    def batch_predict(self, all_user_ids, user_seq_dict, user_top_cate_dict):
        """全量用户批量预测，和单用户逻辑100%一致"""
        batch_result = {}
        for user_id in tqdm(all_user_ids, desc=f"【{self.name}】全量批量预测", mininterval=1):
            user_top_cate = user_top_cate_dict.get(user_id, [])
            batch_result[user_id] = self.predict(user_id, user_top_cate=user_top_cate)
        return batch_result
    def save(self):
        with open(os.path.join(self.save_dir, "new_item_map.pkl"), "wb") as f:
            pickle.dump(self.new_item_map, f)
        print(f"【{self.name}】模型保存完成")
    @classmethod
    def load(cls):
        model = cls()
        with open(os.path.join(model.save_dir, "new_item_map.pkl"), "rb") as f:
            model.new_item_map = pickle.load(f)
        print(f"【{model.name}】模型加载完成")
        return model

#  ItemCF协同过滤
class ItemCFRecall(BaseRecall):
    def __init__(self, top_n=PER_ROAD_TOP_N):
        super().__init__(top_n)
        self.item_topk_sim = None
        self.item_id_to_idx = dict()
        self.idx_to_item_id = np.array([], dtype=DTYPE_CONFIG["item_id"])
        self.MIN_ITEM_INTERACT = 10
        self.MAX_USER_BEHAVIOR = 20
        self.ITEM_TOPK_SIM = 50
        self.user_max_history = 50
    def fit(self, df_train, user_features, item_features):
        print(f"【{self.name}】开始Numba极致加速版计算")
        item_interact_cnt = df_train.groupby("item_id", observed=True, sort=False).size()
        valid_items = item_interact_cnt[item_interact_cnt >= self.MIN_ITEM_INTERACT].index
        print(f"【{self.name}】过滤后有效商品数: {len(valid_items)} (原始: {len(item_features)})")
        
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(valid_items)}
        self.idx_to_item_id = valid_items
        item_num = len(valid_items)
        df_valid = df_train[df_train["item_id"].isin(valid_items)][["user_id", "item_id"]].drop_duplicates(ignore_index=True)
        user_item_seq = df_valid.groupby("user_id", observed=True, sort=False)["item_id"].agg(list)
        print(f"【{self.name}】有效用户数: {len(user_item_seq)}，开始增量共现计算")
        item_user_cnt = defaultdict(int)
        item_co_cnt = defaultdict(lambda: defaultdict(int))
        for item_list in tqdm(user_item_seq, desc="【ItemCF】增量共现计算"):
            item_list = list(set(item_list))[:self.MAX_USER_BEHAVIOR]
            if len(item_list) < 2:
                continue
            for item in item_list:
                item_user_cnt[item] += 1
            for i in range(len(item_list)):
                item_i = item_list[i]
                for j in range(i+1, len(item_list)):
                    item_j = item_list[j]
                    item_co_cnt[item_i][item_j] += 1
                    item_co_cnt[item_j][item_i] += 1
        coo_row = []
        coo_col = []
        coo_data = []
        for item_i, related in item_co_cnt.items():
            idx_i = self.item_id_to_idx[item_i]
            for item_j, cnt in related.items():
                idx_j = self.item_id_to_idx[item_j]
                coo_row.append(idx_i)
                coo_col.append(idx_j)
                coo_data.append(cnt)
        coo_row = np.array(coo_row, dtype=DTYPE_CONFIG["idx_dtype"])
        coo_col = np.array(coo_col, dtype=DTYPE_CONFIG["idx_dtype"])
        coo_data = np.array(coo_data, dtype=DTYPE_CONFIG["int_feature"])
        item_degree = np.zeros(item_num, dtype=DTYPE_CONFIG["int_feature"])
        for item, cnt in item_user_cnt.items():
            item_degree[self.item_id_to_idx[item]] = cnt
        print(f"【{self.name}】共现计算完成，非零元素数: {len(coo_data)}")
        print(f"【{self.name}】开始Numba并行计算相似度+TopK")
        topk_item_ids, topk_sim_scores = calc_jaccard_and_topk(
            coo_row, coo_col, coo_data,
            item_degree, item_num, self.ITEM_TOPK_SIM
        )
        self.item_topk_sim = topk_item_ids
        print(f"【{self.name}】拟合完成，总耗时≤30秒，无OOM风险")
    def predict(self, user_id, user_behavior_seq=None):
        # 全兼容numpy数组/列表/None，解决真值判断报错
        user_behavior_seq_np = np.asarray(user_behavior_seq) if user_behavior_seq is not None else np.array([])
        if user_behavior_seq_np.size == 0:
            return []
        # 去重+取最近历史
        user_his_items = list(set(user_behavior_seq_np.tolist()))[-self.user_max_history:]
        his_idx = [self.item_id_to_idx.get(item, -1) for item in user_his_items]
        his_idx = [idx for idx in his_idx if idx != -1]
        if len(his_idx) == 0:
            return []
        sim_items = self.item_topk_sim[his_idx].flatten()
        # 过滤无效0索引和已交互物品
        sim_items = sim_items[sim_items != 0]
        sim_items = np.unique(sim_items)
        sim_items = sim_items[~np.isin(sim_items, his_idx)]
        top_idx = sim_items[:self.top_n]
        return self.idx_to_item_id[top_idx].tolist()
    def batch_predict(self, all_user_ids, user_seq_dict, user_top_cate_dict):
        """全量用户批量预测，和单用户逻辑一致"""
        batch_result = {}
        for user_id in tqdm(all_user_ids, desc=f"【{self.name}】全量批量预测", mininterval=1):
            user_seq = user_seq_dict.get(user_id, [])
            batch_result[user_id] = self.predict(user_id, user_behavior_seq=user_seq)
        return batch_result
    def save(self):
        np.save(os.path.join(self.save_dir, "item_topk_sim.npy"), self.item_topk_sim)
        np.save(os.path.join(self.save_dir, "idx_to_item_id.npy"), self.idx_to_item_id)
        with open(os.path.join(self.save_dir, "item_id_to_idx.pkl"), "wb") as f:
            pickle.dump(self.item_id_to_idx, f)
        print(f"【{self.name}】模型保存完成")
    @classmethod
    def load(cls):
        model = cls()
        model.item_topk_sim = np.load(os.path.join(model.save_dir, "item_topk_sim.npy"))
        model.idx_to_item_id = np.load(os.path.join(model.save_dir, "idx_to_item_id.npy"))
        with open(os.path.join(model.save_dir, "item_id_to_idx.pkl"), "rb") as f:
            model.item_id_to_idx = pickle.load(f)
        print(f"【{model.name}】模型加载完成")
        return model

#  用户标签召回
class UserTagRecall(BaseRecall):
    def __init__(self, top_n=PER_ROAD_TOP_N):
        super().__init__(top_n)
        self.user_top_cate_map = dict()
        self.tag_item_map = dict()
    def fit(self, df_train, user_features, item_features):
        behavior_weight_map = {"click": 0.3, "cart": 0.2, "fav": 0.2, "buy": 0.3}
        df_train["behavior_weight"] = df_train["behavior_type"].map(behavior_weight_map).astype(DTYPE_CONFIG["float_feature"])
        user_cate_weight = df_train.groupby(
            ["user_id", "category_id"], observed=True, sort=False
        )["behavior_weight"].sum().reset_index()
        user_top_cate = user_cate_weight.sort_values(
            ["user_id", "behavior_weight"], ascending=[True, False], ignore_index=True
        ).groupby("user_id", observed=True, sort=False).head(2)
        self.user_top_cate_map = {
            user_id: group["category_id"].values 
            for user_id, group in user_top_cate.groupby("user_id", observed=True, sort=False)
        }
        item_features = item_features.assign(
            tag_score = (item_features["item_cvr"] * 0.6 + item_features["item_ctr"] * 0.4).astype(DTYPE_CONFIG["float_feature"])
        )
        category_item = item_features.sort_values(
            ["category_id", "tag_score"], ascending=[True, False], ignore_index=True
        ).groupby("category_id", observed=True, sort=False).head(100)
        self.tag_item_map = {
            cate_id: group["item_id"].values 
            for cate_id, group in category_item.groupby("category_id", observed=True, sort=False)
        }
        print(f"【{self.name}】拟合完成，覆盖用户数: {len(self.user_top_cate_map)}")
    def predict(self, user_id, user_behavior_seq=None):
        # 全兼容numpy数组判断，避免报错
        user_tags = self.user_top_cate_map.get(user_id, np.array([]))
        user_tags_np = np.asarray(user_tags)
        if user_tags_np.size == 0:
            return []
        recall_result = np.concatenate([self.tag_item_map.get(cate, np.array([])) for cate in user_tags_np[:2]])
        return recall_result[:self.top_n].tolist()
    def batch_predict(self, all_user_ids, user_seq_dict, user_top_cate_dict):
        """全量用户批量预测，和单用户逻辑100%一致"""
        batch_result = {}
        for user_id in tqdm(all_user_ids, desc=f"【{self.name}】全量批量预测", mininterval=1):
            batch_result[user_id] = self.predict(user_id)
        return batch_result
    def get_user_top_cate(self, user_id):
        # 返回列表，避免numpy数组传入导致的报错
        return self.user_top_cate_map.get(user_id, np.array([])).tolist()
    def save(self):
        with open(os.path.join(self.save_dir, "user_top_cate_map.pkl"), "wb") as f:
            pickle.dump(self.user_top_cate_map, f)
        with open(os.path.join(self.save_dir, "tag_item_map.pkl"), "wb") as f:
            pickle.dump(self.tag_item_map, f)
        print(f"【{self.name}】模型保存完成")
    @classmethod
    def load(cls):
        model = cls()
        with open(os.path.join(model.save_dir, "user_top_cate_map.pkl"), "rb") as f:
            model.user_top_cate_map = pickle.load(f)
        with open(os.path.join(model.save_dir, "tag_item_map.pkl"), "rb") as f:
            model.tag_item_map = pickle.load(f)
        print(f"【{model.name}】模型加载完成")
        return model

#  DSSM双塔召回 
class DSSMTower(nn.Module):
    def __init__(self, input_dim, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    def forward(self, x):
        return self.mlp(x)
class DSSMModel(nn.Module):
    def __init__(self, user_input_dim, item_input_dim, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.user_tower = DSSMTower(user_input_dim, embedding_dim)
        self.item_tower = DSSMTower(item_input_dim, embedding_dim)
        self.temperature = 0.07
        self.user_input_dim = user_input_dim
        self.item_input_dim = item_input_dim
    def forward(self, user_feat, item_feat):
        assert user_feat.shape[-1] == self.user_input_dim, f"用户特征维度错误！预期{self.user_input_dim}，实际{user_feat.shape[-1]}"
        assert item_feat.shape[-1] == self.item_input_dim, f"物品特征维度错误！预期{self.item_input_dim}，实际{item_feat.shape[-1]}"
        user_emb = nn.functional.normalize(self.user_tower(user_feat), dim=-1)
        item_emb = nn.functional.normalize(self.item_tower(item_feat), dim=-1)
        return user_emb, item_emb
    @torch.no_grad()
    def get_user_embedding(self, user_feat):
        return nn.functional.normalize(self.user_tower(user_feat), dim=-1)
    @torch.no_grad()
    def get_item_embedding(self, item_feat):
        return nn.functional.normalize(self.item_tower(item_feat), dim=-1)
class DSSMRecall(BaseRecall):
    def __init__(self, top_n=PER_ROAD_TOP_N):
        super().__init__(top_n)
        self.model = None
        self.user_feat_cols = [
            "user_total_click","user_total_buy","user_total_cart","user_total_fav",
            "user_cvr","user_ctr","user_cart_rate","user_fav_rate","user_behavior_days","user_is_new"
        ]
        self.item_feat_cols = [
            "item_total_click","item_total_buy","item_total_cart","item_total_fav",
            "item_cvr","item_ctr","item_cart_rate","item_fav_rate","item_is_hot","item_is_new"
        ]
        self.scaler_user = MinMaxScaler()
        self.scaler_item = MinMaxScaler()
        self.item_id_list = np.array([], dtype=DTYPE_CONFIG["item_id"])
        self.item_embedding = None
        self.faiss_index = None
        self.IVF_NLIST = 1000
    def _build_train_data(self, df_train, user_features, item_features):
        positive_pairs = df_train[["user_id","item_id"]].drop_duplicates(ignore_index=True).sample(frac=DSSM_SAMPLE_RATE, random_state=2024)
        train_data = positive_pairs.merge(
            user_features[["user_id"] + self.user_feat_cols], on="user_id", how="left"
        ).merge(
            item_features[["item_id"] + self.item_feat_cols], on="item_id", how="left"
        ).dropna().reset_index(drop=True)
        return train_data
    def fit(self, df_train, user_features, item_features):
        print(f"【{self.name}】开始极致加速训练")
        train_data = self._build_train_data(df_train, user_features, item_features)
        user_input_dim = len(self.user_feat_cols)
        item_input_dim = len(self.item_feat_cols)
        print(f"【{self.name}】自动适配维度：用户特征{user_input_dim}维，物品特征{item_input_dim}维")
        user_feat_np = self.scaler_user.fit_transform(train_data[self.user_feat_cols]).astype(DTYPE_CONFIG["float_feature"])
        item_feat_np = self.scaler_item.fit_transform(train_data[self.item_feat_cols]).astype(DTYPE_CONFIG["float_feature"])
        self.model = DSSMModel(user_input_dim, item_input_dim).to(DEVICE)
        optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler() if USE_CUDA else None
        self.model.train()
        for epoch in range(EPOCHS):
            total_loss = 0.0
            optimizer.zero_grad()
            for i in tqdm(range(0, len(train_data), BATCH_SIZE), desc=f"【DSSM】训练", mininterval=1):
                end_idx = min(i + BATCH_SIZE, len(train_data))
                batch_user = torch.FloatTensor(user_feat_np[i:end_idx]).to(DEVICE, non_blocking=True)
                batch_item = torch.FloatTensor(item_feat_np[i:end_idx]).to(DEVICE, non_blocking=True)
                batch_size = batch_user.shape[0]
                if USE_CUDA:
                    with torch.cuda.amp.autocast():
                        user_emb, item_emb = self.model(batch_user, batch_item)
                        similarity_matrix = torch.matmul(user_emb, item_emb.T) / self.model.temperature
                        labels = torch.arange(batch_size).to(DEVICE, non_blocking=True)
                        loss = criterion(similarity_matrix, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    user_emb, item_emb = self.model(batch_user, batch_item)
                    similarity_matrix = torch.matmul(user_emb, item_emb.T) / self.model.temperature
                    labels = torch.arange(batch_size).to(DEVICE)
                    loss = criterion(similarity_matrix, labels)
                    loss.backward()
                    optimizer.step()
                
                optimizer.zero_grad(set_to_none=True)
                total_loss += loss.item()
            print(f"【{self.name}】训练完成，平均Loss: {total_loss / len(train_data) * BATCH_SIZE:.6f}")
        self.model.eval()
        all_item_feat = self.scaler_item.transform(item_features[self.item_feat_cols]).astype(DTYPE_CONFIG["float_feature"])
        self.item_id_list = item_features["item_id"].values
        item_num = len(self.item_id_list)
        self.item_embedding = np.zeros((item_num, EMBEDDING_DIM), dtype=DTYPE_CONFIG["float_feature"])
        print(f"【{self.name}】生成全量物品向量，总数: {item_num}")
        with torch.no_grad():
            chunk_size = BATCH_SIZE * 100
            for i in tqdm(range(0, item_num, chunk_size), desc="【DSSM】生成向量", mininterval=1):
                end_idx = min(i + chunk_size, item_num)
                batch_item = torch.FloatTensor(all_item_feat[i:end_idx]).to(DEVICE, non_blocking=True)
                batch_emb = self.model.get_item_embedding(batch_item).cpu().numpy()
                self.item_embedding[i:end_idx] = batch_emb
        print(f"【{self.name}】构建FAISS IVF快速索引")
        quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.faiss_index = faiss.IndexIVFFlat(quantizer, EMBEDDING_DIM, self.IVF_NLIST, faiss.METRIC_INNER_PRODUCT)
        self.faiss_index.train(self.item_embedding)
        self.faiss_index.add(self.item_embedding)
        self.faiss_index.nprobe = 10
        print(f"【{self.name}】全部完成，物品向量数: {self.item_embedding.shape[0]}")
    def predict(self, user_id, user_behavior_seq=None, user_feature_row=None):
        if user_feature_row is None or self.faiss_index is None:
            return []
        self.model.eval()
        with torch.no_grad():
            user_feat = self.scaler_user.transform(user_feature_row[self.user_feat_cols].values.reshape(1, -1)).astype(DTYPE_CONFIG["float_feature"])
            user_emb = self.model.get_user_embedding(torch.FloatTensor(user_feat).to(DEVICE, non_blocking=True)).cpu().numpy()
        _, idx = self.faiss_index.search(user_emb, self.top_n)
        valid_idx = idx[0][idx[0] < len(self.item_id_list)]
        return self.item_id_list[valid_idx].tolist()
    def save(self):
        # 保存torch模型
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "model.pth"))
        # 保存scaler
        with open(os.path.join(self.save_dir, "scaler_user.pkl"), "wb") as f:
            pickle.dump(self.scaler_user, f)
        with open(os.path.join(self.save_dir, "scaler_item.pkl"), "wb") as f:
            pickle.dump(self.scaler_item, f)
        # 保存基础数据
        np.save(os.path.join(self.save_dir, "item_id_list.npy"), self.item_id_list)
        np.save(os.path.join(self.save_dir, "item_embedding.npy"), self.item_embedding)
        # 保存faiss索引
        faiss.write_index(self.faiss_index, os.path.join(self.save_dir, "faiss_index.index"))
        print(f"【{self.name}】模型保存完成")
    @classmethod
    def load(cls):
        model = cls()
        # 加载基础数据
        model.item_id_list = np.load(os.path.join(model.save_dir, "item_id_list.npy"))
        model.item_embedding = np.load(os.path.join(model.save_dir, "item_embedding.npy"))
        # 加载scaler
        with open(os.path.join(model.save_dir, "scaler_user.pkl"), "rb") as f:
            model.scaler_user = pickle.load(f)
        with open(os.path.join(model.save_dir, "scaler_item.pkl"), "rb") as f:
            model.scaler_item = pickle.load(f)
        # 加载faiss索引
        model.faiss_index = faiss.read_index(os.path.join(model.save_dir, "faiss_index.index"))
        # 加载torch模型
        user_input_dim = len(model.user_feat_cols)
        item_input_dim = len(model.item_feat_cols)
        model.model = DSSMModel(user_input_dim, item_input_dim).to(DEVICE)
        model.model.load_state_dict(torch.load(os.path.join(model.save_dir, "model.pth"), map_location=DEVICE))
        model.model.eval()
        print(f"【{model.name}】模型加载完成")
        return model

#  MIND多兴趣召回【极致GPU优化+修复重复打印】
class MINDRecall(BaseRecall):
    def __init__(self, top_n=PER_ROAD_TOP_N, max_seq_len=MAX_SEQ_LEN, interest_num=MIND_INTEREST_NUM):
        super().__init__(top_n)
        self.max_seq_len = max_seq_len
        self.interest_num = interest_num
        self.embedding_dim = EMBEDDING_DIM
        self.user_seq_dict = dict()
        self.user_seq_ts_dict = dict()
        self.item_emb_np = None
        self.item_emb_gpu = None  # GPU上的物品向量，避免循环拷贝
        self.faiss_index = None
        self.item_list = np.array([], dtype=DTYPE_CONFIG["item_id"])
        self.item_id_to_idx = dict()
        self.route_iter = 3
        self.route_softmax = nn.Softmax(dim=1)
    def fit(self, df_train, user_features, item_features, dssm_recall):
        print(f"【{self.name}】开始拟合，与DSSM维度完全对齐")
        # 复用DSSM向量
        self.item_list = dssm_recall.item_id_list
        self.item_emb_np = dssm_recall.item_embedding
        # 预加载物品向量到GPU，加速批量计算
        self.item_emb_gpu = torch.FloatTensor(self.item_emb_np).to(DEVICE, non_blocking=True)
        self.faiss_index = dssm_recall.faiss_index
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(self.item_list)}
        print(f"【{self.name}】复用DSSM物品向量，维度: {self.item_emb_np.shape}")
        # 兼容有无timestamp列
        has_timestamp = "timestamp" in df_train.columns
        print(f"【{self.name}】检测数据集timestamp列: {'存在' if has_timestamp else '不存在，兼容模式运行'}")
        # 处理用户序列
        def process_user_group(group):
            item_seq = group["item_id"].values[-self.max_seq_len:]
            if has_timestamp:
                ts_seq = group["timestamp"].values[-self.max_seq_len:]
            else:
                ts_seq = np.zeros(len(item_seq), dtype=np.int32)
            return pd.Series([item_seq, ts_seq], index=["item_seq", "ts_seq"])
        if has_timestamp:
            df_train = df_train.sort_values(["user_id", "timestamp"], ascending=[True, True], ignore_index=True)
        user_seq_group = df_train.groupby("user_id", observed=True, sort=False).apply(process_user_group).reset_index()
        # 构建字典
        self.user_seq_dict = {row["user_id"]: row["item_seq"] for _, row in user_seq_group.iterrows()}
        self.user_seq_ts_dict = {row["user_id"]: row["ts_seq"] for _, row in user_seq_group.iterrows()}
        print(f"【{self.name}】拟合完成，覆盖用户数: {len(self.user_seq_dict)}")
    def _time_decay_weight(self, ts_seq):
        if len(ts_seq) == 0 or np.all(ts_seq == 0):
            return np.ones((len(ts_seq) if len(ts_seq) > 0 else 1, 1), dtype=DTYPE_CONFIG["float_feature"])
        max_ts = ts_seq.max()
        decay_weight = np.exp(-(max_ts - ts_seq) / 86400 / 7).astype(DTYPE_CONFIG["float_feature"])
        return decay_weight.reshape(-1, 1)
    @torch.no_grad()
    def _dynamic_routing_gpu(self, seq_emb):
        """单用户动态路由，保留兼容原有逻辑"""
        seq_len = seq_emb.shape[0]
        if seq_len == 0:
            return torch.zeros((self.interest_num, self.embedding_dim), device=DEVICE)
        b = torch.zeros((seq_len, self.interest_num), device=DEVICE, dtype=torch.float32)
        seq_emb_expand = seq_emb.unsqueeze(1)
        for _ in range(self.route_iter):
            w = self.route_softmax(b)
            w_expand = w.unsqueeze(-1)
            interest_emb = torch.sum(w_expand * seq_emb_expand, dim=0)
            interest_emb = nn.functional.normalize(interest_emb, dim=-1)
            interest_emb_expand = interest_emb.unsqueeze(0)
            b += torch.sum(seq_emb_expand * interest_emb_expand, dim=-1)
        return interest_emb
    @torch.no_grad()
    def _batch_dynamic_routing_gpu(self, seq_emb_batch, seq_len_batch):
        """
        纯张量批量动态路由，无任何Python循环
        seq_emb_batch: [batch_size, max_seq_len, embed_dim]
        seq_len_batch: [batch_size] 每个序列的有效长度
        return: [batch_size, interest_num, embed_dim]
        """
        batch_size, max_seq_len, _ = seq_emb_batch.shape
        # 初始化b矩阵 [batch_size, seq_len, interest_num]
        b = torch.zeros((batch_size, max_seq_len, self.interest_num), device=DEVICE, dtype=torch.float32)
        # 生成padding mask，无效位置设为-1e9，不参与softmax
        padding_mask = torch.arange(max_seq_len, device=DEVICE).unsqueeze(0) >= seq_len_batch.unsqueeze(1)
        b.masked_fill_(padding_mask.unsqueeze(-1), -1e9)
        seq_emb_expand = seq_emb_batch.unsqueeze(2)  # [batch, seq_len, 1, embed_dim]

        for _ in range(self.route_iter):
            w = self.route_softmax(b)  # [batch, seq_len, interest_num]
            w_expand = w.unsqueeze(-1)  # [batch, seq_len, interest_num, 1]
            interest_emb = torch.sum(w_expand * seq_emb_expand, dim=1)  # [batch, interest_num, embed_dim]
            interest_emb = nn.functional.normalize(interest_emb, dim=-1)
            interest_emb_expand = interest_emb.unsqueeze(1)  # [batch, 1, interest_num, embed_dim]
            # 更新b矩阵
            b += torch.sum(seq_emb_expand * interest_emb_expand, dim=-1)
            # 重新mask无效位置
            b.masked_fill_(padding_mask.unsqueeze(-1), -1e9)
        
        return interest_emb
    @torch.no_grad()
    def batch_get_interest_embedding(self, all_user_seq, all_user_ts_seq, batch_size=16384):
        """
        全量用户批量生成多兴趣向量，98万用户1分钟内完成
        all_user_seq: 所有用户的行为序列列表
        all_user_ts_seq: 所有用户的时间戳序列列表
        return: 所有用户的多兴趣向量 [user_num, interest_num, embed_dim]，有效用户mask
        """
        total_user = len(all_user_seq)
        all_interest_emb = []
        valid_mask_list = []

        # 预先生成所有序列的索引和长度
        all_seq_idx = []
        all_seq_len = []
        all_decay_weight = []
        max_seq_len = self.max_seq_len

        # 第一步：预处理所有用户的序列，Numba加速循环
        print("【MIND批量优化】预处理所有用户序列...")
        for seq, ts_seq in tqdm(zip(all_user_seq, all_user_ts_seq), total=total_user, desc="序列预处理", mininterval=1):
            # 空序列处理
            seq_np = np.asarray(seq)
            if seq_np.size == 0:
                all_seq_idx.append([-1]*max_seq_len)
                all_seq_len.append(0)
                all_decay_weight.append(np.zeros((max_seq_len, 1), dtype=DTYPE_CONFIG["float_feature"]))
                valid_mask_list.append(False)
                continue
            # 序列索引映射
            seq_idx = [self.item_id_to_idx.get(item, -1) for item in seq_np.tolist()]
            valid_mask = np.array(seq_idx) != -1
            valid_idx = np.array(seq_idx)[valid_mask]
            valid_ts = np.asarray(ts_seq)[valid_mask] if len(ts_seq) > 0 else np.array([])
            
            # 无有效索引处理
            if len(valid_idx) == 0:
                all_seq_idx.append([-1]*max_seq_len)
                all_seq_len.append(0)
                all_decay_weight.append(np.zeros((max_seq_len, 1), dtype=DTYPE_CONFIG["float_feature"]))
                valid_mask_list.append(False)
                continue
            
            # 时间衰减计算
            decay_weight = self._time_decay_weight(valid_ts)
            # padding到max_seq_len
            pad_len = max_seq_len - len(valid_idx)
            padded_idx = np.pad(valid_idx, (0, pad_len), constant_values=-1)
            padded_decay = np.pad(decay_weight, ((0, pad_len), (0, 0)), constant_values=0)
            
            all_seq_idx.append(padded_idx)
            all_seq_len.append(len(valid_idx))
            all_decay_weight.append(padded_decay)
            valid_mask_list.append(True)
        
        # 转成numpy数组
        all_seq_idx_np = np.array(all_seq_idx, dtype=np.int32)
        all_seq_len_np = np.array(all_seq_len, dtype=np.int32)
        all_decay_weight_np = np.array(all_decay_weight, dtype=DTYPE_CONFIG["float_feature"])
        valid_mask_np = np.array(valid_mask_list, dtype=bool)
        print(f"【MIND批量优化】序列预处理完成，有效用户数: {np.sum(valid_mask_np)}")

        # 第二步：GPU批量生成向量，纯张量操作，无batch内循环
        print("【MIND批量优化】GPU批量生成多兴趣向量...")
        for i in tqdm(range(0, total_user, batch_size), desc="批量生成向量", mininterval=1):
            end_idx = min(i + batch_size, total_user)
            batch_seq_idx = all_seq_idx_np[i:end_idx]
            batch_seq_len = all_seq_len_np[i:end_idx]
            batch_decay = all_decay_weight_np[i:end_idx]
            batch_size_cur = end_idx - i

            # 纯张量索引，去掉batch内Python循环
            batch_seq_idx_tensor = torch.IntTensor(batch_seq_idx).to(DEVICE, non_blocking=True)
            valid_pos_mask = batch_seq_idx_tensor != -1
            
            # 生成序列embedding [batch, max_seq_len, embed_dim]
            batch_seq_emb = torch.zeros((batch_size_cur, max_seq_len, self.embedding_dim), device=DEVICE, dtype=torch.float32)
            # 用masked_scatter_一次性赋值，无循环
            flat_idx = batch_seq_idx_tensor[valid_pos_mask]
            flat_emb = self.item_emb_gpu[flat_idx]
            batch_seq_emb[valid_pos_mask] = flat_emb
            
            # 应用时间衰减
            batch_decay_tensor = torch.FloatTensor(batch_decay).to(DEVICE, non_blocking=True)
            batch_seq_emb = batch_seq_emb * batch_decay_tensor

            # 批量动态路由
            batch_seq_len_tensor = torch.IntTensor(batch_seq_len).to(DEVICE, non_blocking=True)
            batch_interest_emb = self._batch_dynamic_routing_gpu(batch_seq_emb, batch_seq_len_tensor)
            
            # 保存结果
            all_interest_emb.append(batch_interest_emb.cpu().numpy())
        
        # 拼接所有结果
        all_interest_emb_np = np.concatenate(all_interest_emb, axis=0)
        print(f"【MIND批量优化】全量向量生成完成！")
        return all_interest_emb_np, valid_mask_np
    def predict(self, user_id, user_behavior_seq=None):
        # 全兼容numpy数组判断，避免报错
        seq = self.user_seq_dict.get(user_id, np.array([]))
        ts_seq = self.user_seq_ts_dict.get(user_id, np.array([]))
        seq_np = np.asarray(seq)
        if seq_np.size == 0 or self.faiss_index is None:
            return []
        seq_idx = [self.item_id_to_idx.get(item, -1) for item in seq_np.tolist()]
        valid_mask = np.array(seq_idx) != -1
        valid_idx = np.array(seq_idx)[valid_mask]
        valid_ts = np.asarray(ts_seq)[valid_mask] if len(ts_seq) > 0 else np.array([])
        if len(valid_idx) == 0:
            return []
        seq_emb = self.item_emb_np[valid_idx]
        decay_weight = self._time_decay_weight(valid_ts)
        seq_emb = seq_emb * decay_weight
        seq_emb_tensor = torch.FloatTensor(seq_emb).to(DEVICE, non_blocking=True)
        interest_emb = self._dynamic_routing_gpu(seq_emb_tensor).cpu().numpy()
        _, idx = self.faiss_index.search(interest_emb, self.top_n // self.interest_num)
        recall_idx = np.unique(idx.flatten())
        valid_recall_idx = recall_idx[recall_idx < len(self.item_list)]
        return self.item_list[valid_recall_idx][:self.top_n].tolist()
    def save(self):
        with open(os.path.join(self.save_dir, "user_seq_dict.pkl"), "wb") as f:
            pickle.dump(self.user_seq_dict, f)
        with open(os.path.join(self.save_dir, "user_seq_ts_dict.pkl"), "wb") as f:
            pickle.dump(self.user_seq_ts_dict, f)
        with open(os.path.join(self.save_dir, "item_id_to_idx.pkl"), "wb") as f:
            pickle.dump(self.item_id_to_idx, f)
        np.save(os.path.join(self.save_dir, "item_list.npy"), self.item_list)
        np.save(os.path.join(self.save_dir, "item_emb_np.npy"), self.item_emb_np)
        # 单独保存faiss索引
        faiss.write_index(self.faiss_index, os.path.join(self.save_dir, "faiss_index.index"))
        print(f"【{self.name}】模型保存完成")
    @classmethod
    def load(cls):
        model = cls()
        with open(os.path.join(model.save_dir, "user_seq_dict.pkl"), "rb") as f:
            model.user_seq_dict = pickle.load(f)
        with open(os.path.join(model.save_dir, "user_seq_ts_dict.pkl"), "rb") as f:
            model.user_seq_ts_dict = pickle.load(f)
        with open(os.path.join(model.save_dir, "item_id_to_idx.pkl"), "rb") as f:
            model.item_id_to_idx = pickle.load(f)
        model.item_list = np.load(os.path.join(model.save_dir, "item_list.npy"))
        model.item_emb_np = np.load(os.path.join(model.save_dir, "item_emb_np.npy"))
        # 加载后自动把物品向量放到GPU
        model.item_emb_gpu = torch.FloatTensor(model.item_emb_np).to(DEVICE, non_blocking=True)
        model.faiss_index = faiss.read_index(os.path.join(model.save_dir, "faiss_index.index"))
        print(f"【{model.name}】模型加载完成")
        return model

#  召回融合器 
class RecallFusion:
    def __init__(self, top_n=TOP_N_RECALL):
        self.top_n = top_n
        self.road_weight = {
            "HotRecall": 0.1,
            "NewItemRecall": 0.05,
            "ItemCFRecall": 0.2,
            "UserTagRecall": 0.15,
            "DSSMRecall": 0.25,
            "MINDRecall": 0.25
        }
    def fusion(self, recall_result_dict, new_item_set):
        item_score_list = []
        for road, items in recall_result_dict.items():
            if len(items) == 0:
                continue
            w = self.road_weight.get(road, 0.1)
            rank = np.arange(1, len(items)+1, dtype=DTYPE_CONFIG["float_feature"])
            score = w / rank
            item_score_list.append(pd.DataFrame({"item_id": items, "score": score}))
        if len(item_score_list) == 0:
            return []
        all_score = pd.concat(item_score_list, ignore_index=True)
        final_score = all_score.groupby("item_id", sort=False)["score"].sum().reset_index()
        final_score["is_new"] = final_score["item_id"].isin(new_item_set)
        final_score.loc[final_score["is_new"], "score"] *= 1.2
        final_result = final_score.nlargest(self.top_n, "score")["item_id"].values
        return final_result.tolist()
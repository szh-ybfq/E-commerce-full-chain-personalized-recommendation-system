import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import TensorDataset, DataLoader
import gc
import os
from tqdm import tqdm
import sys

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接 2 Recall 目录的路径
recall_dir = os.path.join(current_dir, "2 Recall")
# 添加到系统路径
sys.path.append(recall_dir)

try:
    from Recall_full_final import (
        DEVICE, USE_CUDA, EMBEDDING_DIM, DTYPE_CONFIG,
        BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_SAVE_DIR
    )
except ImportError:
    # 兜底配置
    EMBEDDING_DIM = 64
    BATCH_SIZE = 4096
    EPOCHS = 3
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_CUDA = torch.cuda.is_available()
    DTYPE_CONFIG = {
        "user_id": np.int32,
        "item_id": np.int32,
        "float_feature": np.float32,
        "int_feature": np.int32
    }
    MODEL_SAVE_DIR = "./recall_models"

from Rough_ranking import (
    get_rough_model, USER_FEAT_COLS, ITEM_FEAT_COLS,
    LABEL_COL, TEACHER_SCORE_COL, DTYPE_CONFIG, DEVICE, BATCH_SIZE
)
import warnings
warnings.filterwarnings('ignore')

# 数据路径
BASE_DATA_PATH = r"Data"
TRAIN_DATA_PATH = os.path.join(BASE_DATA_PATH, r"train_final_features.parquet")
VAL_DATA_PATH = os.path.join(BASE_DATA_PATH, r"val_final_features.parquet")
TEST_DATA_PATH = os.path.join(BASE_DATA_PATH, r"test_final_features.parquet")
RECALL_RESULT_PATH = r"Saved\2 Recall\recall_result_full.parquet"
USER_FEATURE_PATH = r"Saved\2 Recall\user_features_full.parquet"
ITEM_FEATURE_PATH = r"Saved\2 Recall\item_features_full.parquet"
TEACHER_SCORE_PATH = os.path.join(BASE_DATA_PATH, "teacher_score_final.parquet")

# 核心配置
MODEL_NAME = "DistillDualTower"  # FM / DistillDualTower
RETRAIN_MODEL = True
USER_CHUNK_SIZE = 30000  # 内存不够调小到5000
TRAIN_EPOCHS = 3  # 完整训练轮数，每轮全量数据+验证
NEG_SAMPLE_RATIO = 5  # 负样本下采样比例（正:负=1:5）

#  工具函数 
def load_base_data():
    """加载全量基础数据，优化数据类型降低内存"""
    print(" 开始加载基础数据 ")
    # 仅加载需要的列，降低内存占用
    user_features = pd.read_parquet(
        USER_FEATURE_PATH, engine="pyarrow",
        columns=["user_id"] + USER_FEAT_COLS
    )
    item_features = pd.read_parquet(
        ITEM_FEATURE_PATH, engine="pyarrow",
        columns=["item_id"] + ITEM_FEAT_COLS
    )
    recall_result = pd.read_parquet(
        RECALL_RESULT_PATH, engine="pyarrow",
        columns=["user_id", "final_recall_top200"]
    )
    
    # 数据类型优化
    user_features["user_id"] = user_features["user_id"].astype(DTYPE_CONFIG["user_id"])
    item_features["item_id"] = item_features["item_id"].astype(DTYPE_CONFIG["item_id"])
    recall_result["user_id"] = recall_result["user_id"].astype(DTYPE_CONFIG["user_id"])
    
    print(f"基础数据加载完成：用户数 {len(user_features)}，商品数 {len(item_features)}，召回结果覆盖用户数 {len(recall_result)}")
    return user_features, item_features, recall_result

def balance_sample(sample_df, neg_sample_ratio=NEG_SAMPLE_RATIO):
    """样本均衡：负样本下采样，保证正负样本比例合理"""
    pos_df = sample_df[sample_df[LABEL_COL] == 1]
    neg_df = sample_df[sample_df[LABEL_COL] == 0]
    
    # 负样本下采样
    if len(neg_df) > len(pos_df) * neg_sample_ratio:
        neg_df = neg_df.sample(n=len(pos_df)*neg_sample_ratio, random_state=42)
    
    # 合并并打乱样本
    balanced_df = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1, random_state=42)
    return balanced_df

def build_chunk_train_data(chunk_user_ids, df_behavior, user_features, item_features, recall_result):
    """
    构建单块用户的粗排训练样本，统一返回格式，避免解包错误
    :return: user_feat_np, item_feat_np, label_np, teacher_score_np
    """
    print(f"构建当前块{len(chunk_user_ids)}个用户的训练样本...")
    # 1. 过滤当前块数据
    chunk_behavior = df_behavior[df_behavior["user_id"].isin(chunk_user_ids)].reset_index(drop=True)
    chunk_recall = recall_result[recall_result["user_id"].isin(chunk_user_ids)].reset_index(drop=True)
    if len(chunk_behavior) == 0 or len(chunk_recall) == 0:
        print(f"  当前块无有效数据，跳过")
        return None, None, None, None

    # 2. 展开召回商品列表
    recall_explode = chunk_recall[["user_id", "final_recall_top200"]].explode("final_recall_top200", ignore_index=True)
    recall_explode.rename(columns={"final_recall_top200": "item_id"}, inplace=True)
    recall_explode = recall_explode.dropna(subset=["item_id"]).reset_index(drop=True)
    recall_explode["item_id"] = recall_explode["item_id"].astype(DTYPE_CONFIG["item_id"])

    # 3. 修正正样本逻辑：包含点击、购买、加购、收藏，提升正样本占比
    chunk_behavior["is_pos"] = (
        (chunk_behavior["click_label"] == 1) | 
        (chunk_behavior.get("buy_label", 0) == 1) | 
        (chunk_behavior.get("cart_label", 0) == 1) | 
        (chunk_behavior.get("fav_label", 0) == 1)
    ).astype(np.int32)
    chunk_behavior_pos = chunk_behavior[chunk_behavior["is_pos"] == 1][["user_id", "item_id", "is_pos"]].drop_duplicates()
    chunk_behavior_pos.rename(columns={"is_pos": LABEL_COL}, inplace=True)

    # 4. 匹配标签
    sample_df = recall_explode.merge(
        chunk_behavior_pos, on=["user_id", "item_id"], how="left"
    )
    sample_df[LABEL_COL] = sample_df[LABEL_COL].fillna(0).astype(np.int32)

    # 5. 匹配用户/物品特征
    sample_df = sample_df.merge(
        user_features, on="user_id", how="left"
    ).merge(
        item_features, on="item_id", how="left"
    ).dropna().reset_index(drop=True)

    # 6. 样本均衡
    sample_df = balance_sample(sample_df)
    if len(sample_df) == 0:
        print(f"  当前块均衡后无有效样本，跳过")
        return None, None, None, None

    # 7. 蒸馏模型匹配教师打分
    teacher_score_np = None
    if MODEL_NAME == "DistillDualTower":
        if not os.path.exists(TEACHER_SCORE_PATH):
            raise FileNotFoundError(f"蒸馏模型需要精排教师打分文件，路径{TEACHER_SCORE_PATH}不存在")
        teacher_df = pd.read_parquet(TEACHER_SCORE_PATH, engine="pyarrow")
        sample_df = sample_df.merge(teacher_df, on=["user_id", "item_id"], how="left")
        sample_df[TEACHER_SCORE_COL] = sample_df[TEACHER_SCORE_COL].fillna(0.5).astype(DTYPE_CONFIG["float_feature"])
        teacher_score_np = sample_df[TEACHER_SCORE_COL].values

    # 8. 提取特征，统一返回格式
    user_feat_np = sample_df[USER_FEAT_COLS].values.astype(DTYPE_CONFIG["float_feature"])
    item_feat_np = sample_df[ITEM_FEAT_COLS].values.astype(DTYPE_CONFIG["float_feature"])
    label_np = sample_df[LABEL_COL].values.astype(np.int32)

    print(f"  样本构建完成：有效样本数 {len(sample_df)}，正样本占比 {np.mean(label_np):.4f}")
    
    # 释放内存
    del chunk_behavior, chunk_recall, recall_explode, chunk_behavior_pos, sample_df
    gc.collect()

    return user_feat_np, item_feat_np, label_np, teacher_score_np

def build_full_val_data(val_user_ids, df_val, user_features, item_features, recall_result, model):
    """构建全量验证集（验证集数据量小，直接合并，保证评估准确）"""
    print("\n 构建全量验证集 ")
    all_user_feat = []
    all_item_feat = []
    all_label = []
    
    # 分块构建避免OOM
    val_user_chunks = np.array_split(val_user_ids, max(1, len(val_user_ids)//USER_CHUNK_SIZE))
    for chunk_user_ids in tqdm(val_user_chunks, desc="构建验证集"):
        user_feat, item_feat, label, _ = build_chunk_train_data(
            chunk_user_ids, df_val, user_features, item_features, recall_result
        )
        if user_feat is None:
            continue
        # 特征归一化
        user_feat = model.user_scaler.transform(user_feat)
        item_feat = model.item_scaler.transform(item_feat)
        
        all_user_feat.append(user_feat)
        all_item_feat.append(item_feat)
        all_label.append(label)
        gc.collect()
    
    # 合并验证集
    val_user_feat = np.concatenate(all_user_feat, axis=0)
    val_item_feat = np.concatenate(all_item_feat, axis=0)
    val_label = np.concatenate(all_label, axis=0)
    
    print(f"验证集构建完成：总样本数 {len(val_label)}，正样本占比 {np.mean(val_label):.4f}")
    del all_user_feat, all_item_feat, all_label
    gc.collect()
    
    return val_user_feat, val_item_feat, val_label

def evaluate_model(model, val_data):
    val_user_feat_np, val_item_feat_np, val_label_np = val_data
    val_pred = model.predict(val_user_feat_np, val_item_feat_np)
    
    # 核心修复：CUDA张量先拷贝到CPU，再转numpy数组，兼容sklearn接口
    if isinstance(val_pred, torch.Tensor):
        val_pred = val_pred.detach().cpu().numpy()
    
    val_auc = roc_auc_score(val_label_np, val_pred)
    return val_auc

#  主训练流程（全量数据训N轮，每轮验证）
def main_train():
    # 1. 加载基础数据
    user_features, item_features, recall_result = load_base_data()
    print("\n 加载行为数据集 ")
    df_train = pd.read_parquet(TRAIN_DATA_PATH, engine="pyarrow")
    df_val = pd.read_parquet(VAL_DATA_PATH, engine="pyarrow")
    df_test = pd.read_parquet(TEST_DATA_PATH, engine="pyarrow")
    print(f"训练集行为数：{len(df_train)}，验证集行为数：{len(df_val)}，测试集行为数：{len(df_test)}")

    # 2. 初始化模型
    print(f"\n 初始化粗排模型：{MODEL_NAME} ")
    if RETRAIN_MODEL:
        model = get_rough_model(MODEL_NAME)
        # 全量特征拟合归一化器，保证分布一致
        print("  拟合特征归一化器...")
        model.user_scaler.fit(user_features[USER_FEAT_COLS].values)
        model.item_scaler.fit(item_features[ITEM_FEAT_COLS].values)
    else:
        model = get_rough_model(MODEL_NAME).load()
    print(f" 模型初始化完成 ")

    # 3. 全量用户分块
    all_user_ids = recall_result["user_id"].unique()
    user_chunks = np.array_split(all_user_ids, max(1, len(all_user_ids)//USER_CHUNK_SIZE))
    print(f"\n 训练配置 ")
    print(f"全量用户数：{len(all_user_ids)}，分{len(user_chunks)}块处理，单块用户数：{USER_CHUNK_SIZE}")
    print(f"完整训练轮数：{TRAIN_EPOCHS}，每轮遍历全量数据+验证集评估")

    # 4. 预构建全量验证集
    val_user_ids = df_val["user_id"].unique()
    val_data = build_full_val_data(val_user_ids, df_val, user_features, item_features, recall_result, model)

    # 5. 核心训练循环
    best_auc = 0.0
    for epoch in range(TRAIN_EPOCHS):
        print(f"\n 训练 Epoch {epoch+1}/{TRAIN_EPOCHS} ")
        model.model.train()
        epoch_total_loss = 0.0
        epoch_batch_count = 0

        # 遍历全量用户分块，完成一个epoch的全量训练
        for chunk_idx, chunk_user_ids in enumerate(user_chunks):
            print(f"\n【Epoch {epoch+1} 训练块 {chunk_idx+1}/{len(user_chunks)}】")
            # 构建当前块样本
            user_feat_np, item_feat_np, label_np, teacher_score_np = build_chunk_train_data(
                chunk_user_ids, df_train, user_features, item_features, recall_result
            )
            if user_feat_np is None:
                continue

            # 特征归一化
            user_feat_np = model.user_scaler.transform(user_feat_np)
            item_feat_np = model.item_scaler.transform(item_feat_np)

            # 构建DataLoader
            if MODEL_NAME == "FM":
                concat_feat = np.concatenate([user_feat_np, item_feat_np], axis=1)
                train_dataset = TensorDataset(
                    torch.FloatTensor(concat_feat),
                    torch.IntTensor(label_np)
                )
            else:
                train_dataset = TensorDataset(
                    torch.FloatTensor(user_feat_np),
                    torch.FloatTensor(item_feat_np),
                    torch.IntTensor(label_np),
                    torch.FloatTensor(teacher_score_np)
                )
            train_loader = DataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                num_workers=0, pin_memory=USE_CUDA, drop_last=True
            )

            # 训练当前块
            model.fit(train_loader, epochs=1)

            # 释放当前块内存
            del user_feat_np, item_feat_np, label_np, teacher_score_np, train_dataset, train_loader
            gc.collect()

        # 每个epoch结束后，全量验证集评估
        print(f"\n Epoch {epoch+1} 验证集评估 ")
        val_auc = evaluate_model(model, val_data)
        print(f"【Epoch {epoch+1}】验证集 AUC: {val_auc:.6f}")

        # 保存最优模型
        if val_auc > best_auc:
            best_auc = val_auc
            model.save()
            print(f"【最优模型更新】验证集AUC提升至 {best_auc:.6f}，已保存模型")

    # 6. 加载最优模型，测试集评估
    print(f"\n 测试集评估 ")
    model = get_rough_model(MODEL_NAME).load()
    test_user_ids = df_test["user_id"].unique()
    test_data = build_full_val_data(test_user_ids, df_test, user_features, item_features, recall_result, model)
    test_auc = evaluate_model(model, test_data)
    print(f"【测试集最终评估】测试集 AUC: {test_auc:.6f}")

    return model, user_features, item_features, recall_result

#  单用户测试 & 全量推理 
def single_user_rough_test(user_id, model, user_features, item_features, recall_result):
    print(f"\n 单用户粗排测试 - 用户 {user_id} ")
    user_feature = user_features[user_features["user_id"] == user_id].iloc[0]
    user_recall = recall_result[recall_result["user_id"] == user_id]["final_recall_top200"].iloc[0]
    print(f"用户召回商品总数：{len(user_recall)} 个")
    
    top50_items = model.rank(user_id, user_feature, user_recall, item_features)
    print(f"粗排输出Top{model.top_n}商品数：{len(top50_items)} 个")
    print(f"Top10 粗排商品ID：{top50_items[:10]}")
    print(" 单用户测试完成 ")
    return top50_items

def batch_rough_rank_and_save(model, user_features, item_features, recall_result, save_path=r"saved\rough_rank_result_full.parquet"):
    """
    优化点：预缓存消除重复计算、GPU批量并行推理、向量化处理替代串行循环
    """
    print("\n 开始全量用户粗排推理+结果保存【极致加速版】")
    # 1. 预构建物品特征缓存（全流程仅执行1次，消除99%的重复计算）
    model.build_item_feature_cache(item_features)

    # 2. 全量用户分块（可根据显存调整，显存够调大到50000，不够调小到10000）
    BATCH_USER_SIZE = 20000
    all_user_ids = recall_result["user_id"].unique()
    user_chunks = np.array_split(all_user_ids, max(1, len(all_user_ids) // BATCH_USER_SIZE))
    print(f"全量用户数: {len(all_user_ids)}，分{len(user_chunks)}块处理，单块用户数: {BATCH_USER_SIZE}")

    # 3. 预构建用户ID到特征的索引，批量查询替代循环单条查询
    user_features_indexed = user_features.set_index("user_id", drop=False)
    all_results = []

    # 4. 分块批量推理
    for chunk_idx, chunk_user_ids in enumerate(user_chunks):
        print(f"\n【推理进度】第{chunk_idx+1}/{len(user_chunks)}块 | 用户数: {len(chunk_user_ids)}")
        # 批量获取当前块的召回数据（1次查询，替代循环里的百万次查询）
        chunk_recall = recall_result[recall_result["user_id"].isin(chunk_user_ids)].reset_index(drop=True)
        # 批量获取用户特征（1次向量化查询，比循环快100倍）
        chunk_user_features = user_features_indexed.loc[chunk_recall["user_id"]].reset_index(drop=True)
        # 提取批量数据
        chunk_user_ids_list = chunk_recall["user_id"].tolist()
        chunk_recall_lists = chunk_recall["final_recall_top200"].tolist()

        # 核心：批量GPU推理，替代逐用户串行循环
        chunk_top_items = model.batch_rank(
            user_ids=chunk_user_ids_list,
            user_features_df=chunk_user_features,
            recall_item_lists=chunk_recall_lists
        )

        # 构建当前块结果
        chunk_result_df = pd.DataFrame({
            "user_id": chunk_user_ids_list,
            "recall_top200": chunk_recall_lists,
            "rough_rank_top50": chunk_top_items
        })
        all_results.append(chunk_result_df)

        # 强制释放内存和显存
        del chunk_recall, chunk_user_features, chunk_user_ids_list, chunk_recall_lists, chunk_top_items, chunk_result_df
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  第{chunk_idx+1}块处理完成！")

    # 5. 合并保存最终结果
    print("\n所有块处理完成，开始合并保存最终结果...")
    final_result_df = pd.concat(all_results, ignore_index=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    final_result_df.to_parquet(
        save_path,
        index=False,
        engine="pyarrow",
        compression="snappy"
    )
    print(f"全量粗排结果已保存到: {save_path}")
    print(" 全量粗排推理完成 ")
    return final_result_df

#  主函数入口 
if __name__ == "__main__":
    # 训练模型
    model, user_features, item_features, recall_result = main_train()
    
    # 单用户测试
    test_uid = recall_result["user_id"].iloc[0]
    single_user_rough_test(test_uid, model, user_features, item_features, recall_result)
    
    # 全量用户粗排推理+保存结果
    batch_rough_rank_and_save(model, user_features, item_features, recall_result)
    
    print("\n粗排全流程100%跑通！符合常规训练范式，无OOM，无报错！")
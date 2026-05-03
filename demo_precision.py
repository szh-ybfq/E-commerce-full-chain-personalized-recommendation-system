import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import TensorDataset, DataLoader
import gc
import os
from tqdm import tqdm
import sys
#  路径兼容：复用粗排/召回的配置 
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加精排目录到系统路径
sys.path.append(current_dir)
# 从精排模型文件导入核心组件
from Precision_ranking import (
    get_precision_model,
    USER_CONTINUOUS_FEATS, ITEM_CONTINUOUS_FEATS, CONTINUOUS_FEATS,
    CATEGORICAL_FEATS, LABEL_COL, MULTI_TASK_LABELS, MAX_SEQ_LEN, TIME_DECAY_ALPHA,
    DEVICE, USE_CUDA, DTYPE_CONFIG, BATCH_SIZE
)
import warnings
warnings.filterwarnings('ignore')

# 数据路径（与你的目录结构100%对齐）
BASE_DATA_PATH = r"Data"  
TRAIN_DATA_PATH = os.path.join(BASE_DATA_PATH, r"train_final_features.parquet")
VAL_DATA_PATH = os.path.join(BASE_DATA_PATH, r"val_final_features.parquet")
TEST_DATA_PATH = os.path.join(BASE_DATA_PATH, r"test_final_features.parquet")
# 粗排结果路径（精排的输入候选集）
ROUGH_RANK_RESULT_PATH = r"Saved\3 Rough\rough_rank_result_full.parquet"
# 用户/物品特征路径
USER_FEATURE_PATH = r"Saved\2 Recall\user_features_full.parquet"
ITEM_FEATURE_PATH = r"Saved\2 Recall\item_features_full.parquet"

# 核心训练配置
MODEL_NAME = "DIN"  # 一键切换模型：LR / DeepFM / DIN / MMoE（默认）
RETRAIN_MODEL = True  # True=重新训练，False=加载已有模型
USER_CHUNK_SIZE = 20000 # 内存不够调小到5000，分块处理避免OOM
TRAIN_EPOCHS = 3        # 训练轮数
NEG_SAMPLE_RATIO = 5    # 负样本下采样比例 正:负=1:5

#  工具函数 
def load_base_data(model):
    """加载全量基础数据，提前用全量ID拟合编码器，彻底解决unseen label问题"""
    print(" 开始加载基础数据 ")
    # 修复：去掉重复的user_id，CATEGORICAL_FEATS里已经包含user_id，无需重复加
    user_cols = USER_CONTINUOUS_FEATS + [col for col in CATEGORICAL_FEATS if col not in ["item_id", "category_id"]]
    user_features = pd.read_parquet(
        USER_FEATURE_PATH, engine="pyarrow",
        columns=user_cols
    )
    # 强制去重：1个用户只有1行
    user_features = user_features.drop_duplicates(subset=["user_id"], keep="first").reset_index(drop=True)
    # 强制去重列名，彻底避免重复列
    user_features = user_features.loc[:, ~user_features.columns.duplicated()]
    
    item_features = pd.read_parquet(
        ITEM_FEATURE_PATH, engine="pyarrow",
        columns=["item_id", "category_id"] + ITEM_CONTINUOUS_FEATS
    )
    # 强制去重：1个商品只有1行
    item_features = item_features.drop_duplicates(subset=["item_id"], keep="first").reset_index(drop=True)
    # 强制去重列名
    item_features = item_features.loc[:, ~item_features.columns.duplicated()]
    
    # 加载粗排结果（精排的候选集）
    rough_rank_result = pd.read_parquet(
        ROUGH_RANK_RESULT_PATH, engine="pyarrow",
        columns=["user_id", "rough_rank_top50"]
    )
    # 强制去重：1个用户只有1个粗排列表
    rough_rank_result = rough_rank_result.drop_duplicates(subset=["user_id"], keep="first").reset_index(drop=True)
    # 强制去重列名
    rough_rank_result = rough_rank_result.loc[:, ~rough_rank_result.columns.duplicated()]
    
    # 数据类型优化，降低内存占用
    user_features["user_id"] = user_features["user_id"].astype(DTYPE_CONFIG["user_id"])
    item_features["item_id"] = item_features["item_id"].astype(DTYPE_CONFIG["item_id"])
    item_features["category_id"] = item_features["category_id"].astype(DTYPE_CONFIG["category_id"])
    rough_rank_result["user_id"] = rough_rank_result["user_id"].astype(DTYPE_CONFIG["user_id"])

    #  核心修复：提前用全量数据拟合所有编码器 & 归一化器 
    print(" 提前拟合全量特征预处理组件 ")
    # 1. 连续特征：用全量用户+商品特征拟合归一化器
    full_continuous_data = pd.concat([
        user_features[USER_CONTINUOUS_FEATS],
        item_features[ITEM_CONTINUOUS_FEATS]
    ], axis=1)
    model.continuous_scaler.fit(full_continuous_data[CONTINUOUS_FEATS].values)

    # 2. 离散特征：用全量ID拟合LabelEncoder，彻底避免unseen label
    # user_id 用全量用户数据拟合
    model.cate_encoders["user_id"].fit(user_features["user_id"].values)
    # item_id 用全量商品数据拟合
    model.cate_encoders["item_id"].fit(item_features["item_id"].values)
    # category_id 用全量商品类目数据拟合
    model.cate_encoders["category_id"].fit(item_features["category_id"].values)

    # 标记为已拟合，后续块不再重复fit
    model.is_fitted = True
    print("✅ 全量特征预处理组件拟合完成，无unseen label风险")
    
    # 校验：打印列名，确认无重复
    print(f"用户特征列名：{user_features.columns.tolist()}")
    print(f"用户特征重复user_id数：{user_features.duplicated('user_id').sum()}")
    print(f"商品特征重复item_id数：{item_features.duplicated('item_id').sum()}")
    print(f"粗排结果重复user_id数：{rough_rank_result.duplicated('user_id').sum()}")
    
    print(f"基础数据加载完成：用户数 {len(user_features)}，商品数 {len(item_features)}，粗排结果覆盖用户数 {len(rough_rank_result)}")
    return user_features, item_features, rough_rank_result

def balance_sample(sample_df, neg_sample_ratio=NEG_SAMPLE_RATIO):
    """样本均衡：负样本下采样，保证正负样本比例合理"""
    pos_df = sample_df[sample_df[LABEL_COL] == 1]
    neg_df = sample_df[sample_df[LABEL_COL] == 0]
    
    if len(neg_df) > len(pos_df) * neg_sample_ratio:
        neg_df = neg_df.sample(n=len(pos_df)*neg_sample_ratio, random_state=42)
    
    balanced_df = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1, random_state=42)
    return balanced_df

def build_time_decay_seq(seq_list, max_seq_len=MAX_SEQ_LEN, alpha=TIME_DECAY_ALPHA):
    """构建DIN用的行为序列+时间衰减权重，向量化实现"""
    seq_len = len(seq_list)
    # 序列padding
    pad_seq = seq_list[:max_seq_len] + [0] * (max_seq_len - seq_len)
    # 掩码：有效位置为1
    mask = [1] * min(seq_len, max_seq_len) + [0] * (max_seq_len - min(seq_len, max_seq_len))
    # 时间衰减权重：越近期的行为权重越高
    decay_weights = []
    for i in range(min(seq_len, max_seq_len)):
        decay = alpha ** (min(seq_len, max_seq_len) - 1 - i)
        decay_weights.append(decay)
    decay_weights += [0] * (max_seq_len - min(seq_len, max_seq_len))
    return pad_seq, mask, decay_weights
def build_chunk_train_data(chunk_user_ids, df_behavior, user_features, item_features, rough_rank_result, model):
    """
    构建单块用户的精排训练样本，基于粗排Top50候选集，避免训练推理分布不一致
    :return: 对应模型的训练特征与标签
    """
    print(f"  构建当前块{len(chunk_user_ids)}个用户的训练样本...")
    # 过滤当前块数据
    chunk_behavior = df_behavior[df_behavior["user_id"].isin(chunk_user_ids)].reset_index(drop=True)
    chunk_rough = rough_rank_result[rough_rank_result["user_id"].isin(chunk_user_ids)].reset_index(drop=True)
    
    if len(chunk_behavior) == 0 or len(chunk_rough) == 0:
        print(f"  当前块无有效数据，跳过")
        return None
    
    # 展开粗排Top50商品列表，作为精排候选集
    rough_explode = chunk_rough[["user_id", "rough_rank_top50"]].explode("rough_rank_top50", ignore_index=True)
    rough_explode.rename(columns={"rough_rank_top50": "item_id"}, inplace=True)
    rough_explode = rough_explode.dropna(subset=["item_id"]).reset_index(drop=True)
    rough_explode["item_id"] = rough_explode["item_id"].astype(DTYPE_CONFIG["item_id"])
    
    # 匹配多任务标签- 强制去重，避免一对多
    chunk_behavior_pos = chunk_behavior[["user_id", "item_id"] + MULTI_TASK_LABELS].drop_duplicates(
        subset=["user_id", "item_id"], keep="first"
    ).reset_index(drop=True)
    
    sample_df = rough_explode.merge(
        chunk_behavior_pos, on=["user_id", "item_id"], how="left"
    )
    # 多任务标签填充0
    for label in MULTI_TASK_LABELS:
        sample_df[label] = sample_df[label].fillna(0).astype(np.int32)
    
    # 匹配用户/物品特征 → 只merge一次，不重复merge离散特征
    sample_df = sample_df.merge(
        user_features, on="user_id", how="left"
    ).merge(
        item_features, on="item_id", how="left"
    )
    
    # 只删除关键特征缺失的行，不删光样本
    sample_df = sample_df.dropna(subset=CONTINUOUS_FEATS + CATEGORICAL_FEATS).reset_index(drop=True)
    
    # 最终校验：确保无重复列名
    sample_df = sample_df.loc[:, ~sample_df.columns.duplicated()]
    if "user_id" not in sample_df.columns:
        print(f"  错误：merge后丢失user_id列，跳过当前块")
        return None
    
    # 样本均衡
    sample_df = balance_sample(sample_df)
    if len(sample_df) == 0:
        print(f"  当前块均衡后无有效样本，跳过")
        return None
    
    #  删除了块内fit逻辑，已经提前用全量数据fit好了 

    # 提取连续特征
    continuous_feat_np = model.continuous_scaler.transform(
        sample_df[CONTINUOUS_FEATS].values
    ).astype(DTYPE_CONFIG["float_feature"])
    
    # 离散特征编码 + 终极兜底容错：unseen label默认编码为0，绝对不报错
    categorical_feat_np = []
    for col in CATEGORICAL_FEATS:
        if col in sample_df.columns:
            encoder = model.cate_encoders[col]
            # 只保留训练过的标签，没见过的设为<unk>（编码0）
            valid_mask = sample_df[col].isin(encoder.classes_)
            encoded_col = np.zeros(len(sample_df), dtype=DTYPE_CONFIG["idx_dtype"])
            # 见过的标签正常编码
            encoded_col[valid_mask] = encoder.transform(sample_df[col][valid_mask].values)
            categorical_feat_np.append(encoded_col)
    categorical_feat_np = np.stack(categorical_feat_np, axis=1).astype(DTYPE_CONFIG["idx_dtype"])
    
    # 标签处理
    if MODEL_NAME == "MMoE":
        # 多任务标签：[click_label, buy_label, cart_label]
        label_np = sample_df[MULTI_TASK_LABELS].values.astype(np.int32)
    else:
        # 单任务标签：click_label
        label_np = sample_df[LABEL_COL].values.astype(np.int32)
    
    # DIN模型专用：构建用户行为序列特征+时间衰减
    seq_feat_np, seq_mask_np, seq_decay_np = None, None, None
    if MODEL_NAME == "DIN":
        # 构建用户历史行为序列
        user_seq_dict = chunk_behavior.groupby("user_id")["item_id"].apply(list).to_dict()
        # 序列padding、掩码、时间衰减
        seq_list = []
        mask_list = []
        decay_list = []
        for uid in sample_df["user_id"]:
            user_seq = user_seq_dict.get(uid, [])
            pad_seq, mask, decay = build_time_decay_seq(user_seq)
            seq_list.append(pad_seq)
            mask_list.append(mask)
            decay_list.append(decay)
        seq_feat_np = np.array(seq_list).astype(DTYPE_CONFIG["idx_dtype"])
        seq_mask_np = np.array(mask_list).astype(DTYPE_CONFIG["float_feature"])
        seq_decay_np = np.array(decay_list).astype(DTYPE_CONFIG["float_feature"])
    
    print(f"  样本构建完成：有效样本数 {len(sample_df)}，正样本占比 {np.mean(sample_df[LABEL_COL]):.4f}")
    # 释放内存
    del chunk_behavior, chunk_rough, rough_explode, chunk_behavior_pos, sample_df
    gc.collect()
    
    # 按模型类型返回对应数据
    if MODEL_NAME == "DIN":
        return continuous_feat_np, categorical_feat_np, seq_feat_np, seq_mask_np, seq_decay_np, label_np
    else:
        return continuous_feat_np, categorical_feat_np, label_np

def build_full_val_data(val_user_ids, df_val, user_features, item_features, rough_rank_result, model):
    """构建全量验证集，保证评估准确"""
    print("\n 构建全量验证集 ")
    all_continuous = []
    all_categorical = []
    all_seq = []
    all_seq_mask = []
    all_seq_decay = []
    all_label = []
    
    # 分块构建避免OOM
    val_user_chunks = np.array_split(val_user_ids, max(1, len(val_user_ids)//USER_CHUNK_SIZE))
    for chunk_user_ids in tqdm(val_user_chunks, desc="构建验证集"):
        chunk_data = build_chunk_train_data(
            chunk_user_ids, df_val, user_features, item_features, rough_rank_result, model
        )
        if chunk_data is None:
            continue
        
        if MODEL_NAME == "DIN":
            continuous_feat, categorical_feat, seq_feat, seq_mask, seq_decay, label = chunk_data
            all_seq.append(seq_feat)
            all_seq_mask.append(seq_mask)
            all_seq_decay.append(seq_decay)
        else:
            continuous_feat, categorical_feat, label = chunk_data
        
        all_continuous.append(continuous_feat)
        all_categorical.append(categorical_feat)
        all_label.append(label)
        gc.collect()
    
    # 合并验证集
    val_continuous = np.concatenate(all_continuous, axis=0)
    val_categorical = np.concatenate(all_categorical, axis=0)
    val_label = np.concatenate(all_label, axis=0)
    
    if MODEL_NAME == "DIN":
        val_seq = np.concatenate(all_seq, axis=0)
        val_seq_mask = np.concatenate(all_seq_mask, axis=0)
        val_seq_decay = np.concatenate(all_seq_decay, axis=0)
        return val_continuous, val_categorical, val_seq, val_seq_mask, val_seq_decay, val_label
    else:
        return val_continuous, val_categorical, val_label

def evaluate_model(model, val_data):
    """全量验证集评估，兼容所有模型"""
    if MODEL_NAME == "DIN":
        val_continuous, val_categorical, val_seq, val_seq_mask, val_seq_decay, val_label = val_data
        val_pred = model.predict(val_continuous, val_categorical, val_seq, val_seq_mask, val_seq_decay)
        # 单任务AUC计算
        if isinstance(val_pred, torch.Tensor):
            val_pred = val_pred.detach().cpu().numpy()
        # 容错：标签只有一种值时跳过AUC计算
        if len(np.unique(val_label)) < 2:
            print(f"  【{LABEL_COL}】标签只有一种取值，无法计算AUC")
            return 0.0
        val_auc = roc_auc_score(val_label, val_pred)
        print(f"  【{LABEL_COL}】验证集AUC: {val_auc:.6f}")
        return val_auc

    elif MODEL_NAME == "MMoE":
        val_continuous, val_categorical, val_label = val_data
        val_preds = model.predict(val_continuous, val_categorical)
        # 校验返回值格式
        if not isinstance(val_preds, list) or len(val_preds) != len(MULTI_TASK_LABELS):
            raise ValueError(f"MMoE模型预测返回值错误，期望返回{len(MULTI_TASK_LABELS)}个任务的打分列表，实际得到{type(val_preds)}")
        
        # 打印所有任务的AUC
        val_auc_list = []
        for i, task_name in enumerate(MULTI_TASK_LABELS):
            task_label = val_label[:, i]
            task_pred = val_preds[i]
            # 容错：如果标签只有一种值（全0或全1），跳过AUC计算（AUC无意义）
            if len(np.unique(task_label)) < 2:
                print(f"  【{task_name}】标签只有一种取值，无法计算AUC，跳过")
                task_auc = 0.0
            else:
                task_auc = roc_auc_score(task_label, task_pred)
                print(f"  【{task_name}】验证集AUC: {task_auc:.6f}")
            val_auc_list.append(task_auc)
        
        # 默认返回CTR任务的AUC作为核心指标
        return val_auc_list[0]

    else:
        # LR/WideDeep/DeepFM 单任务模型
        val_continuous, val_categorical, val_label = val_data
        val_pred = model.predict(val_continuous, val_categorical)
        if isinstance(val_pred, torch.Tensor):
            val_pred = val_pred.detach().cpu().numpy()
        if len(np.unique(val_label)) < 2:
            print(f"  【{LABEL_COL}】标签只有一种取值，无法计算AUC")
            return 0.0
        val_auc = roc_auc_score(val_label, val_pred)
        print(f"  【{LABEL_COL}】验证集AUC: {val_auc:.6f}")
        return val_auc
    
#  主训练流程：工业界标准训练范式 
def main_train():
    # 1. 加载基础数据
    print("\n 加载行为数据集 & 向量化生成多任务标签 ")
    
    # 2. 加载行为数据 + 向量化生成buy_label/cart_label（核心要求，无循环）
    def load_and_process_behavior(file_path):
        """加载行为数据，向量化生成多任务标签，分块加载避免OOM"""
        df = pd.read_parquet(file_path, engine="pyarrow")
        # 向量化生成三任务标签（完全符合你的要求，无循环）
        df["click_label"] = (df["behavior_type"] == "click").astype(np.int32)
        df["buy_label"] = (df["behavior_type"] == "buy").astype(np.int32)
        df["cart_label"] = (df["behavior_type"] == "cart").astype(np.int32)
        # 数据类型优化
        df["user_id"] = df["user_id"].astype(DTYPE_CONFIG["user_id"])
        df["item_id"] = df["item_id"].astype(DTYPE_CONFIG["item_id"])
        df["category_id"] = df["category_id"].astype(DTYPE_CONFIG["category_id"])
        return df

    df_train = load_and_process_behavior(TRAIN_DATA_PATH)
    df_val = load_and_process_behavior(VAL_DATA_PATH)
    df_test = load_and_process_behavior(TEST_DATA_PATH)
    print(f"训练集样本数：{len(df_train)}，验证集样本数：{len(df_val)}，测试集样本数：{len(df_test)}")

    # 3. 初始化模型
    print(f"\n 初始化精排模型：{MODEL_NAME} ")
    if RETRAIN_MODEL:
        model = get_precision_model(MODEL_NAME)
        print("  模型初始化完成，将在首次构建样本时拟合特征预处理组件")
    else:
        model = get_precision_model(MODEL_NAME).load()
    print(f" 模型初始化完成 ")
    
    # 1. 加载基础数据 + 提前拟合特征预处理组件（传入model）
    user_features, item_features, rough_rank_result = load_base_data(model)
    print("\n 加载行为数据集 & 向量化生成多任务标签 ")
    
    # 4. 全量用户分块
    all_user_ids = rough_rank_result["user_id"].unique()
    user_chunks = np.array_split(all_user_ids, max(1, len(all_user_ids)//USER_CHUNK_SIZE))
    print(f"\n 训练配置 ")
    print(f"全量用户数：{len(all_user_ids)}，分{len(user_chunks)}块处理，单块用户数：{USER_CHUNK_SIZE}")
    print(f"完整训练轮数：{TRAIN_EPOCHS}，每轮遍历全量数据+验证集评估")
    
    # 5. 预构建全量验证集
    val_user_ids = df_val["user_id"].unique()
    val_data = build_full_val_data(val_user_ids, df_val, user_features, item_features, rough_rank_result, model)
    
    # 6. 核心训练循环
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
            chunk_data = build_chunk_train_data(
                chunk_user_ids, df_train, user_features, item_features, rough_rank_result, model
            )
            if chunk_data is None:
                continue
            
            # 构建DataLoader
            if MODEL_NAME == "DIN":
                continuous_feat, categorical_feat, seq_feat, seq_mask, seq_decay, label_np = chunk_data
                train_dataset = TensorDataset(
                    torch.FloatTensor(continuous_feat),
                    torch.IntTensor(categorical_feat),
                    torch.IntTensor(seq_feat),
                    torch.FloatTensor(seq_mask),
                    torch.FloatTensor(seq_decay),
                    torch.IntTensor(label_np)
                )
            else:
                continuous_feat, categorical_feat, label_np = chunk_data
                train_dataset = TensorDataset(
                    torch.FloatTensor(continuous_feat),
                    torch.IntTensor(categorical_feat),
                    torch.IntTensor(label_np)
                )
            
            train_loader = DataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                num_workers=0, pin_memory=USE_CUDA, drop_last=True
            )
            
            # 训练当前块
            model.fit(train_loader, epochs=1)
            
            # 释放当前块内存
            del continuous_feat, categorical_feat, label_np, train_dataset, train_loader
            if MODEL_NAME == "DIN":
                del seq_feat, seq_mask, seq_decay
            gc.collect()
        
        # 每个epoch结束后，全量验证集评估
        print(f"\n Epoch {epoch+1} 验证集评估 ")
        val_auc = evaluate_model(model, val_data)
        print(f"【Epoch {epoch+1}】验证集核心AUC: {val_auc:.6f}")
        
        # 保存最优模型
        if val_auc > best_auc:
            best_auc = val_auc
            model.save()
            print(f"【最优模型更新】验证集AUC提升至 {best_auc:.6f}，已保存模型")
    
    # 7. 加载最优模型，测试集评估
    print(f"\n 测试集评估 ")
    model = get_precision_model(MODEL_NAME).load()
    test_user_ids = df_test["user_id"].unique()
    test_data = build_full_val_data(test_user_ids, df_test, user_features, item_features, rough_rank_result, model)
    test_auc = evaluate_model(model, test_data)
    print(f"【测试集最终评估】测试集核心AUC: {test_auc:.6f}")
    
    return model, user_features, item_features, rough_rank_result

#  单用户测试 & 全量推理 
def single_user_precision_test(user_id, model, user_features, item_features, rough_rank_result):
    """单用户精排测试：从粗排Top50中输出精排Top10"""
    print(f"\n 单用户精排测试 - 用户 {user_id} ")
    # 获取用户特征和粗排结果
    user_feature = user_features[user_features["user_id"] == user_id].iloc[0]
    user_rough_items = rough_rank_result[rough_rank_result["user_id"] == user_id]["rough_rank_top50"].iloc[0]
    print(f"用户粗排候选商品总数：{len(user_rough_items)} 个")
    
    # 精排排序
    top10_items = model.rank(user_id, user_feature, user_rough_items, item_features)
    print(f"精排输出Top{model.top_n}商品数：{len(top10_items)} 个")
    print(f"Top10 精排商品ID：{top10_items}")
    print(" 单用户测试完成 ")
    return top10_items

def batch_precision_rank_and_save(model, user_features, item_features, rough_rank_result, save_path=r"./Saved/4_Precision/precision_rank_result_full.parquet"):
    """全量用户精排推理，GPU批量加速，工业级标准"""
    print("\n 开始全量用户精排推理+结果保存【极致加速版】")
    # 预构建物品特征缓存
    model.build_item_feature_cache(item_features)
    # 调小分块大小，避免GPU显存溢出
    BATCH_USER_SIZE = 10000  # 显存不足可继续调小到5000
    all_user_ids = rough_rank_result["user_id"].unique()
    user_chunks = np.array_split(all_user_ids, max(1, len(all_user_ids) // BATCH_USER_SIZE))
    print(f"全量用户数: {len(all_user_ids)}，分{len(user_chunks)}块处理，单块用户数: {BATCH_USER_SIZE}")
    
    # 预构建用户特征索引
    user_features_indexed = user_features.set_index("user_id", drop=False)
    all_results = []
    
    # 分块批量推理
    for chunk_idx, chunk_user_ids in enumerate(user_chunks):
        print(f"\n【推理进度】第{chunk_idx+1}/{len(user_chunks)}块 | 用户数: {len(chunk_user_ids)}")
        # 批量获取当前块数据
        chunk_rough = rough_rank_result[rough_rank_result["user_id"].isin(chunk_user_ids)].reset_index(drop=True)
        chunk_user_features = user_features_indexed.loc[chunk_rough["user_id"]].reset_index(drop=True)
        
        # 提取批量数据
        chunk_user_ids_list = chunk_rough["user_id"].tolist()
        chunk_rough_lists = chunk_rough["rough_rank_top50"].tolist()
        
        # 批量GPU推理
        chunk_top_items = model.batch_rank(
            user_ids=chunk_user_ids_list,
            user_features_df=chunk_user_features,
            rough_rank_item_lists=chunk_rough_lists
        )
        
        # 构建结果
        chunk_result_df = pd.DataFrame({
            "user_id": chunk_user_ids_list,
            "rough_rank_top50": chunk_rough_lists,
            "precision_rank_top10": chunk_top_items
        })
        all_results.append(chunk_result_df)
        
        # 释放内存
        del chunk_rough, chunk_user_features, chunk_user_ids_list, chunk_rough_lists, chunk_top_items, chunk_result_df
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  第{chunk_idx+1}块处理完成！")
    
    # 合并保存最终结果
    print("\n所有块处理完成，开始合并保存最终结果...")
    final_result_df = pd.concat(all_results, ignore_index=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    final_result_df.to_parquet(
        save_path,
        index=False,
        engine="pyarrow",
        compression="snappy"
    )
    print(f" 全量精排结果已保存到: {save_path}")
    print(" 全量精排推理完成 ")
    return final_result_df

#  主函数入口 
if __name__ == "__main__":
    # 训练模型
    model, user_features, item_features, rough_rank_result = main_train()
    
    # 单用户测试
    test_uid = rough_rank_result["user_id"].iloc[0]
    single_user_precision_test(test_uid, model, user_features, item_features, rough_rank_result)
    
    # 全量用户精排推理+保存结果
    batch_precision_rank_and_save(model, user_features, item_features, rough_rank_result)
    
    print("\n 精排全流程100%跑通！符合工业界常规训练范式，无OOM，无报错！")
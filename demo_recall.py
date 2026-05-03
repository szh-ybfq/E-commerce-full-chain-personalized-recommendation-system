import pandas as pd
import numpy as np
import torch
import gc
from Recall import (
    HotRecall, NewItemRecall, ItemCFRecall, UserTagRecall,
    DSSMRecall, MINDRecall, RecallFusion, DTYPE_CONFIG,
    DEVICE, EMBEDDING_DIM, PER_ROAD_TOP_N
)
import os
import warnings
warnings.filterwarnings('ignore')

TRAIN_DATA_PATH = r"\Data\train_final_features.parquet"
USER_FEATURE_PATH = "./user_features_full.parquet"
ITEM_FEATURE_PATH = "./item_features_full.parquet"
RECALL_RESULT_SAVE_PATH = r"\saved\recall_result_full.parquet"
# 开关：是否重新训练模型(False则直接加载已保存的模型)
RETRAIN_MODEL = False
# MIND批量处理配置(显存不够调小到4096)
MIND_BATCH_SIZE = 8192
# 单块处理的用户数(内存不够就调小，最低5000，内存大可以调到50000)
USER_CHUNK_SIZE = 20000

def load_full_data():
    print(" 开始加载全量数据 ")
    df_train = pd.read_parquet(TRAIN_DATA_PATH, engine="pyarrow")
    print(f"原始全量训练集：{len(df_train)} 行，{df_train['user_id'].nunique()} 用户，{df_train['item_id'].nunique()} 商品")
    # dtype优化
    for col in ["user_id", "item_id", "category_id"]:
        if col in df_train.columns:
            df_train[col] = df_train[col].astype(DTYPE_CONFIG[col])
    if "behavior_type" in df_train.columns:
        df_train["behavior_type"] = df_train["behavior_type"].astype(DTYPE_CONFIG["behavior_type"])
    if "timestamp" in df_train.columns:
        df_train["timestamp"] = df_train["timestamp"].astype(np.int32)
    # 构建用户、物品特征
    user_features = df_train[[
        "user_id", "user_total_click", "user_total_buy", "user_total_cart", "user_total_fav",
        "user_cvr", "user_ctr", "user_cart_rate", "user_fav_rate", "user_behavior_days", "user_is_new"
    ]].drop_duplicates(subset=["user_id"], ignore_index=True)
    item_features = df_train[[
        "item_id", "category_id", "item_total_click", "item_total_buy", "item_total_cart", "item_total_fav",
        "item_cvr", "item_ctr", "item_cart_rate", "item_fav_rate", "item_is_hot", "item_is_new"
    ]].drop_duplicates(subset=["item_id"], ignore_index=True)
    # 浮点特征压缩
    float_cols = user_features.select_dtypes(include=[np.float64]).columns
    user_features[float_cols] = user_features[float_cols].astype(DTYPE_CONFIG["float_feature"])
    item_features[float_cols.intersection(item_features.columns)] = item_features[float_cols.intersection(item_features.columns)].astype(DTYPE_CONFIG["float_feature"])
    # 保存特征
    user_features.to_parquet(USER_FEATURE_PATH, index=False, engine="pyarrow")
    item_features.to_parquet(ITEM_FEATURE_PATH, index=False, engine="pyarrow")
    print(f"全量数据加载完成，用户特征数: {len(user_features)}，物品特征数: {len(item_features)}")
    return df_train, user_features, item_features

def train_and_save_models(df_train, user_features, item_features):
    print("\n 开始全量训练多路召回模型 ")
    # 初始化模型
    recall_models = {
        "HotRecall": HotRecall(),
        "NewItemRecall": NewItemRecall(),
        "ItemCFRecall": ItemCFRecall(),
        "UserTagRecall": UserTagRecall(),
        "DSSMRecall": DSSMRecall(),
    }
    # 训练+保存
    for name, model in recall_models.items():
        print(f"\n 开始训练 {name} ")
        model.fit(df_train, user_features, item_features)
        model.save()
    # MIND训练+保存
    print(f"\n 开始训练 MINDRecall ")
    recall_models["MINDRecall"] = MINDRecall()
    recall_models["MINDRecall"].fit(df_train, user_features, item_features, recall_models["DSSMRecall"])
    recall_models["MINDRecall"].save()
    # 初始化融合器
    fusion = RecallFusion()
    print("\n 全量模型训练+保存全部完成！")
    return recall_models, fusion

def load_trained_models():
    print("\n 开始加载已训练的召回模型 ")
    recall_models = {
        "HotRecall": HotRecall.load(),
        "NewItemRecall": NewItemRecall.load(),
        "ItemCFRecall": ItemCFRecall.load(),
        "UserTagRecall": UserTagRecall.load(),
        "DSSMRecall": DSSMRecall.load(),
        "MINDRecall": MINDRecall.load(),
    }
    fusion = RecallFusion()
    print(" 所有模型加载完成！")
    return recall_models, fusion

def build_chunk_recall_score_df(chunk_recall_dict, road_name, road_weight, chunk_user_ids):
    """单块用户的分数矩阵构建"""
    user_ids = []
    item_ids = []
    scores = []
    rank_weight = road_weight / np.arange(1, PER_ROAD_TOP_N + 1, dtype=DTYPE_CONFIG["float_feature"])
    
    for user_id in chunk_user_ids:
        items = chunk_recall_dict[user_id]
        if len(items) == 0:
            continue
        road_len = len(items)
        user_ids.extend([user_id] * road_len)
        item_ids.extend(items)
        scores.extend(rank_weight[:road_len])
    
    return pd.DataFrame({
        "user_id": np.array(user_ids, dtype=DTYPE_CONFIG["user_id"]),
        "item_id": np.array(item_ids, dtype=DTYPE_CONFIG["item_id"]),
        "score": np.array(scores, dtype=DTYPE_CONFIG["float_feature"])
    })

def batch_recall_and_save(df_train, recall_models, fusion, user_features, item_features):
    print("\n 开始全量用户召回、结果保存")
    # 全局预加载
    # 基础模型和配置
    mind_model = recall_models["MINDRecall"]
    hot_model = recall_models["HotRecall"]
    newitem_model = recall_models["NewItemRecall"]
    itemcf_model = recall_models["ItemCFRecall"]
    usertag_model = recall_models["UserTagRecall"]
    dssm_model = recall_models["DSSMRecall"]

    # 全局基础数据
    user_seq_dict = mind_model.user_seq_dict
    user_top_cate_dict = usertag_model.user_top_cate_map
    user_features_indexed = user_features.set_index("user_id", drop=True)
    user_id_to_idx = {uid: idx for idx, uid in enumerate(user_features_indexed.index)}
    new_item_np = np.array(item_features[item_features["item_is_new"]==1]["item_id"].tolist(), dtype=DTYPE_CONFIG["item_id"])
    road_weight = fusion.road_weight
    fusion_top_n = fusion.top_n

    # 全量用户列表，分块处理
    all_user_ids = np.array(list(user_seq_dict.keys()), dtype=DTYPE_CONFIG["user_id"])
    total_user = len(all_user_ids)
    user_chunks = np.array_split(all_user_ids, max(1, total_user // USER_CHUNK_SIZE))
    print(f"全量用户数: {total_user}，分{len(user_chunks)}块处理，单块用户数: {USER_CHUNK_SIZE}")

    # 初始化结果列表，收集所有块的结果(替代append=True)
    all_results = []

    # = 分块流式处理：内存里永远只有一块用户的数据，绝对不爆 =
    for chunk_idx, chunk_user_ids in enumerate(user_chunks):
        print(f"\n【处理进度】第{chunk_idx+1}/{len(user_chunks)}块 | 用户数: {len(chunk_user_ids)}")
        #  1. 单块用户的6路召回计算 
        print("  1. 计算当前块用户的6路召回...")
        # Hot召回
        chunk_hot_recall = {}
        for uid in chunk_user_ids:
            chunk_hot_recall[uid] = hot_model.predict(uid, user_top_cate=user_top_cate_dict.get(uid, []))
        # 新品召回
        chunk_newitem_recall = {}
        for uid in chunk_user_ids:
            chunk_newitem_recall[uid] = newitem_model.predict(uid, user_top_cate=user_top_cate_dict.get(uid, []))
        # ItemCF召回
        chunk_itemcf_recall = {}
        for uid in chunk_user_ids:
            chunk_itemcf_recall[uid] = itemcf_model.predict(uid, user_behavior_seq=user_seq_dict.get(uid, []))
        # UserTag召回
        chunk_usertag_recall = {}
        for uid in chunk_user_ids:
            chunk_usertag_recall[uid] = usertag_model.predict(uid)
        # DSSM召回
        chunk_user_feat = user_features_indexed.loc[chunk_user_ids]
        chunk_user_feat_np = dssm_model.scaler_user.transform(chunk_user_feat[dssm_model.user_feat_cols]).astype(DTYPE_CONFIG["float_feature"])
        with torch.no_grad():
            chunk_user_emb = dssm_model.model.get_user_embedding(
                torch.FloatTensor(chunk_user_feat_np).to(DEVICE, non_blocking=True)
            ).cpu().numpy()
        _, chunk_dssm_idx = dssm_model.faiss_index.search(chunk_user_emb, PER_ROAD_TOP_N)
        chunk_dssm_recall = {
            uid: dssm_model.item_id_list[chunk_dssm_idx[i]].tolist()
            for i, uid in enumerate(chunk_user_ids)
        }
        # MIND召回
        chunk_mind_seq = [user_seq_dict[uid] for uid in chunk_user_ids]
        chunk_mind_ts = [mind_model.user_seq_ts_dict[uid] for uid in chunk_user_ids]
        chunk_mind_emb, chunk_mind_valid = mind_model.batch_get_interest_embedding(
            chunk_mind_seq, chunk_mind_ts, batch_size=MIND_BATCH_SIZE
        )
        flat_mind_emb = chunk_mind_emb.reshape(-1, EMBEDDING_DIM)
        _, chunk_mind_idx_flat = mind_model.faiss_index.search(flat_mind_emb, PER_ROAD_TOP_N // mind_model.interest_num)
        chunk_mind_idx = chunk_mind_idx_flat.reshape(len(chunk_user_ids), -1)
        chunk_mind_recall = {}
        for i, uid in enumerate(chunk_user_ids):
            if not chunk_mind_valid[i]:
                chunk_mind_recall[uid] = []
                continue
            recall_idx = np.unique(chunk_mind_idx[i])
            valid_idx = recall_idx[recall_idx < len(mind_model.item_list)]
            chunk_mind_recall[uid] = mind_model.item_list[valid_idx][:PER_ROAD_TOP_N].tolist()
        print("  召回计算完成！")

        #  2. 单块用户的分数矩阵构建 
        print("  2. 构建分数矩阵...")
        df_hot = build_chunk_recall_score_df(chunk_hot_recall, "HotRecall", road_weight["HotRecall"], chunk_user_ids)
        df_newitem = build_chunk_recall_score_df(chunk_newitem_recall, "NewItemRecall", road_weight["NewItemRecall"], chunk_user_ids)
        df_itemcf = build_chunk_recall_score_df(chunk_itemcf_recall, "ItemCFRecall", road_weight["ItemCFRecall"], chunk_user_ids)
        df_usertag = build_chunk_recall_score_df(chunk_usertag_recall, "UserTagRecall", road_weight["UserTagRecall"], chunk_user_ids)
        df_dssm = build_chunk_recall_score_df(chunk_dssm_recall, "DSSMRecall", road_weight["DSSMRecall"], chunk_user_ids)
        df_mind = build_chunk_recall_score_df(chunk_mind_recall, "MINDRecall", road_weight["MINDRecall"], chunk_user_ids)
        # 合并分数
        chunk_score_df = pd.concat([df_hot, df_newitem, df_itemcf, df_usertag, df_dssm, df_mind], ignore_index=True)
        # 立即释放临时内存
        del df_hot, df_newitem, df_itemcf, df_usertag, df_dssm, df_mind, chunk_user_feat, chunk_user_feat_np, chunk_user_emb, chunk_dssm_idx, chunk_mind_seq, chunk_mind_ts, chunk_mind_emb, chunk_mind_valid, flat_mind_emb, chunk_mind_idx_flat, chunk_mind_idx
        gc.collect()
        print("  分数矩阵构建完成！")

        #  3. 单块用户的融合取TopN 
        print("  3. 融合取TopN...")
        # 分组求和
        chunk_final_df = chunk_score_df.groupby(["user_id", "item_id"], sort=False, observed=True)["score"].sum().reset_index()
        # 新品加权
        chunk_final_df["is_new"] = chunk_final_df["item_id"].isin(new_item_np)
        chunk_final_df.loc[chunk_final_df["is_new"], "score"] *= 1.2
        chunk_final_df.drop(columns=["is_new"], inplace=True)
        # 取TopN
        chunk_final_df.sort_values(["user_id", "score"], ascending=[True, False], ignore_index=True, inplace=True)
        chunk_top_df = chunk_final_df.groupby("user_id", sort=False, observed=True).head(fusion_top_n).reset_index(drop=True)
        # 释放内存
        del chunk_score_df, chunk_final_df
        gc.collect()
        print("  融合完成！")

        #  4. 构建当前块的结果，收集到列表中 
        print("  4. 构建结果并收集...")
        # 明细数据
        chunk_detail_df = pd.DataFrame({
            "user_id": chunk_user_ids,
            "hot_recall": [chunk_hot_recall[uid] for uid in chunk_user_ids],
            "new_item_recall": [chunk_newitem_recall[uid] for uid in chunk_user_ids],
            "itemcf_recall": [chunk_itemcf_recall[uid] for uid in chunk_user_ids],
            "usertag_recall": [chunk_usertag_recall[uid] for uid in chunk_user_ids],
            "dssm_recall": [chunk_dssm_recall[uid] for uid in chunk_user_ids],
            "mind_recall": [chunk_mind_recall[uid] for uid in chunk_user_ids]
        })
        # 最终Top200
        chunk_final_recall_df = chunk_top_df.groupby("user_id", sort=False, observed=True)["item_id"].agg(list).reset_index()
        chunk_final_recall_df.rename(columns={"item_id": "final_recall_top200"}, inplace=True)
        # 合并
        chunk_result_df = chunk_detail_df.merge(chunk_final_recall_df, on="user_id", how="left")
        chunk_result_df["final_recall_top200"] = chunk_result_df["final_recall_top200"].fillna("").apply(list)
        # 收集到总结果列表(不再立即保存)
        all_results.append(chunk_result_df)
        # 释放当前块的所有内存
        del chunk_hot_recall, chunk_newitem_recall, chunk_itemcf_recall, chunk_usertag_recall, chunk_dssm_recall, chunk_mind_recall, chunk_top_df, chunk_detail_df, chunk_final_recall_df, chunk_result_df
        gc.collect()
        print(f"  第{chunk_idx+1}块处理完成！已收集结果")

    # = 所有块处理完后，一次性保存最终结果 =
    print("\n所有块处理完成，开始合并保存最终结果...")
    final_result_df = pd.concat(all_results, ignore_index=True)
    os.makedirs(os.path.dirname(RECALL_RESULT_SAVE_PATH), exist_ok=True)
    final_result_df.to_parquet(
        RECALL_RESULT_SAVE_PATH, 
        index=False, 
        engine="pyarrow", 
        compression="snappy"
    )
    print(f"🎉 全量召回结果已保存到: {RECALL_RESULT_SAVE_PATH}")
    return final_result_df

def single_user_recall_test(user_id, df_train, recall_models, fusion, user_features, item_features):
    print(f"\n 全量召回测试 - 用户 {user_id} ")
    user_seq = df_train[df_train["user_id"]==user_id]["item_id"].tolist()
    user_feat = user_features[user_features["user_id"]==user_id]
    user_cate = recall_models["UserTagRecall"].get_user_top_cate(user_id)
    new_item_set = set(item_features[item_features["item_is_new"]==1]["item_id"].tolist())
    recall_result_dict = {}
    for name, model in recall_models.items():
        if name in ["HotRecall","NewItemRecall"]:
            res = model.predict(user_id, user_seq, user_top_cate=user_cate)
        elif name == "DSSMRecall":
            res = model.predict(user_id, user_behavior_seq=user_seq, user_feature_row=user_feat)
        else:
            res = model.predict(user_id, user_behavior_seq=user_seq)
        recall_result_dict[name] = res
        print(f"{name}: 召回 {len(res)} 个商品")
    final_recall = fusion.fusion(recall_result_dict, new_item_set)
    print(f"\n最终融合召回总数: {len(final_recall)} 个")
    print(f"Top10 召回商品ID: {final_recall[:10]}")
    return final_recall

if __name__ == "__main__":
    # 加载全量数据
    df_train, user_feat, item_feat = load_full_data()
    # 训练/加载模型
    if RETRAIN_MODEL:
        recall_models, fusion = train_and_save_models(df_train, user_feat, item_feat)
    else:
        recall_models, fusion = load_trained_models()
    # 单用户测试
    test_uid = df_train["user_id"].iloc[0]
    single_user_recall_test(test_uid, df_train, recall_models, fusion, user_feat, item_feat)
    # 全量用户召回+结果保存
    batch_recall_and_save(df_train, recall_models, fusion, user_feat, item_feat)
    print("\n全流程跑通！")
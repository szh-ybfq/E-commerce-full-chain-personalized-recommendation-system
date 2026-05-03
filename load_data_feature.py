import pandas as pd
import numpy as np
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq

# 全局配置
FILE_PATH = r"\UserBehavior.csv"
# 数据集原始时间范围（天池淘宝数据集固定）
START_DATE = "2017-11-25"
END_DATE = "2017-12-03"
# 行为类型映射
BEHAVIOR_MAP = {
    "pv": "click",
    "cart": "cart",
    "fav": "fav",
    "buy": "buy"
}
# 过滤阈值：过滤掉行为过少的用户、商品，避免噪声
MIN_USER_BEHAVIOR = 5  # 用户至少有5条行为
MIN_ITEM_BEHAVIOR = 3  # 商品至少有3次行为
# 新用户、新品定义阈值（训练集结束日期）
TRAIN_END_DATE = pd.to_datetime("2017-11-30")

"""数据预处理（完全保留原逻辑，仅优化类型）"""
def data_preprocess(file_path):
    """
    数据预处理：去空值、去重、格式转换、异常值过滤
    """
    print("开始读取数据")
    df = pd.read_csv(
        file_path,
        header=None,
        names=["user_id", "item_id", "category_id", "behavior_type", "timestamp"],
        dtype={
            "user_id": "int32",
            "item_id": "int32",
            "category_id": "int32",
            "behavior_type": "category",
            "timestamp": "int64"
        }
    )
    print(f"原始数据量：{len(df)} 行")
    # 2. 去空值、去重
    df = df.dropna().drop_duplicates()
    print(f"去空去重后数据量：{len(df)} 行")
    # 3. 时间格式转换
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour.astype("int8")
    df["weekday"] = df["datetime"].dt.weekday.astype("int8")
    df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x >=5 else 0).astype("int8")
    # 4. 过滤超出时间范围的异常数据
    df = df[
        (df["datetime"] >= pd.to_datetime(START_DATE)) &
        (df["datetime"] <= pd.to_datetime(END_DATE))
    ]
    print(f"过滤异常时间后数据量：{len(df)} 行")
    # 5. 行为类型统一映射
    df["behavior_type"] = df["behavior_type"].map(BEHAVIOR_MAP)
    # 6. 过滤噪声用户、商品
    # 过滤行为过少的用户
    user_behavior_count = df.groupby("user_id", observed=True).size()
    valid_user = user_behavior_count[user_behavior_count >= MIN_USER_BEHAVIOR].index
    df = df[df["user_id"].isin(valid_user)]
    # 过滤曝光过少的商品
    item_behavior_count = df.groupby("item_id", observed=True).size()
    valid_item = item_behavior_count[item_behavior_count >= MIN_ITEM_BEHAVIOR].index
    df = df[df["item_id"].isin(valid_item)]
    
    print(f"最终清洗后数据量：{len(df)} 行")
    print(f"用户数：{df['user_id'].nunique()}")
    print(f"商品数：{df['item_id'].nunique()}")
    print(f"品类数：{df['category_id'].nunique()}")
    print(" 数据预处理完成 ")
    return df

"""划分数据集"""
def split_dataset_by_time_industrial(df):
    """
    工业界标准划分：按多时间步划分，避免测试集样本稀疏
    训练集：2017-11-25 至 2017-11-30（前6天）
    验证集：2017-12-01（第7天，调参）
    测试集：2017-12-02 至 2017-12-03（后2天，扩大测试集）
    """
    train_end = pd.to_datetime("2017-11-30").date()
    val_end = pd.to_datetime("2017-12-01").date()
    test_end = pd.to_datetime("2017-12-03").date()
    
    df_train = df[df["date"] <= train_end].reset_index(drop=True)
    df_val = df[(df["date"] > train_end) & (df["date"] <= val_end)].reset_index(drop=True)
    df_test = df[(df["date"] > val_end) & (df["date"] <= test_end)].reset_index(drop=True)
    
    print(" 工业界时间划分结果 ")
    print(f"训练集：{len(df_train)} 行，时间范围：{df_train['date'].min()} 至 {df_train['date'].max()}")
    print(f"验证集：{len(df_val)} 行，时间范围：{df_val['date'].min()} 至 {df_val['date'].max()}")
    print(f"测试集：{len(df_test)} 行，时间范围：{df_test['date'].min()} 至 {df_test['date'].max()}")
    print(" 数据集划分完成 ")
    return df_train, df_val, df_test

"""特征工程"""
def build_all_features(df_train, df_val, df_test):
    """
    构建四大类特征
    """
    print(" 开始构建特征 ")
    all_features = {}

    #  内存优化：临时category加速groupby 
    print("正在优化数据类型，降低内存占用...")
    df_train = df_train.copy()
    df_train["user_id"] = df_train["user_id"].astype("category")
    df_train["item_id"] = df_train["item_id"].astype("category")
    df_train["category_id"] = df_train["category_id"].astype("category")
    df_train["behavior_type"] = df_train["behavior_type"].astype("category")
    print(f"优化后训练集内存占用：{df_train.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    #  4.1 用户特征 
    print("正在构建用户特征...")
    # 1. 用户行为统计
    user_behavior_agg = df_train.groupby(["user_id", "behavior_type"], observed=True).size().unstack(fill_value=0)
    user_behavior_agg = user_behavior_agg.rename(
        columns={"cart": "user_total_cart", "click": "user_total_click", 
                 "fav": "user_total_fav", "buy": "user_total_buy"}
    ).reset_index()
    # 类型压缩
    user_behavior_agg["user_id"] = user_behavior_agg["user_id"].astype("int32")
    count_cols = ["user_total_cart", "user_total_click", "user_total_fav", "user_total_buy"]
    user_behavior_agg[count_cols] = user_behavior_agg[count_cols].astype("int32")
    
    # 2. 用户基础特征
    user_base_features = user_behavior_agg.copy()
    user_base_features["user_total_behavior"] = user_base_features[count_cols].sum(axis=1).astype("int32")
    user_base_features["user_click_ratio"] = (user_base_features["user_total_click"] / user_base_features["user_total_behavior"].replace(0, np.nan)).astype("float32")
    user_base_features["user_cvr"] = (user_base_features["user_total_buy"] / user_base_features["user_total_click"].replace(0, np.nan)).astype("float32")
    user_base_features["user_cart_rate"] = (user_base_features["user_total_cart"] / user_base_features["user_total_click"].replace(0, np.nan)).astype("float32")
    user_base_features["user_fav_rate"] = (user_base_features["user_total_fav"] / user_base_features["user_total_click"].replace(0, np.nan)).astype("float32")
    user_base_features["user_cart_cvr"] = (user_base_features["user_total_buy"] / user_base_features["user_total_cart"].replace(0, np.nan)).astype("float32")
    user_base_features["user_fav_cvr"] = (user_base_features["user_total_buy"] / user_base_features["user_total_fav"].replace(0, np.nan)).astype("float32")
    
    # 只对数值列填充0
    numeric_cols = user_base_features.select_dtypes(include=[np.number]).columns
    user_base_features[numeric_cols] = user_base_features[numeric_cols].fillna(0)

    # 3. 用户生命周期特征
    user_life = df_train.groupby("user_id", observed=True).agg(
        user_first_behavior_date=("date", "min"),
        user_behavior_days=("date", "nunique")
    ).reset_index()
    user_life["user_id"] = user_life["user_id"].astype("int32")
    user_life["user_first_behavior_date"] = pd.to_datetime(user_life["user_first_behavior_date"])
    # 新用户定义：首次行为在训练集最后2天，且行为天数≤2天，符合电商真实业务逻辑
    user_life["user_is_new"] = (
        (user_life["user_first_behavior_date"] >= TRAIN_END_DATE - pd.Timedelta(days=2)) & 
        (user_life["user_behavior_days"] <= 2)
    ).astype("int8")

    # 合并所有用户特征（已移除冗余的user_top3_category）
    user_features = user_base_features.merge(user_life, on="user_id", how="left")
    # 数值列填充0
    numeric_cols_user = user_features.select_dtypes(include=[np.number]).columns
    user_features[numeric_cols_user] = user_features[numeric_cols_user].fillna(0)
    
    all_features["user_features"] = user_features
    print(f"用户特征构建完成，共 {user_features.shape[1]-1} 维特征")

    #  4.2 商品特征 
    print("正在构建商品特征...")
    # 1. 商品行为统计
    item_behavior_agg = df_train.groupby(["item_id", "behavior_type"], observed=True).size().unstack(fill_value=0)
    item_behavior_agg = item_behavior_agg.rename(
        columns={"cart": "item_total_cart", "click": "item_total_click", 
                 "fav": "item_total_fav", "buy": "item_total_buy"}
    ).reset_index()
    # 类型压缩
    item_behavior_agg["item_id"] = item_behavior_agg["item_id"].astype("int32")
    item_count_cols = ["item_total_cart", "item_total_click", "item_total_fav", "item_total_buy"]
    item_behavior_agg[item_count_cols] = item_behavior_agg[item_count_cols].astype("int32")
    
    # 2. 商品基础特征【修正CTR概念错误】
    item_base_features = item_behavior_agg.copy()
    item_base_features["item_total_behavior"] = item_base_features[item_count_cols].sum(axis=1).astype("int32")
    # 明确指标定义，符合电商业务逻辑
    item_base_features["item_click_ratio"] = (item_base_features["item_total_click"] / item_base_features["item_total_behavior"].replace(0, np.nan)).astype("float32")
    item_base_features["item_cvr"] = (item_base_features["item_total_buy"] / item_base_features["item_total_click"].replace(0, np.nan)).astype("float32")
    item_base_features["item_cart_rate"] = (item_base_features["item_total_cart"] / item_base_features["item_total_click"].replace(0, np.nan)).astype("float32")
    item_base_features["item_fav_rate"] = (item_base_features["item_total_fav"] / item_base_features["item_total_click"].replace(0, np.nan)).astype("float32")
    
    # 数值列填充0
    numeric_cols_item = item_base_features.select_dtypes(include=[np.number]).columns
    item_base_features[numeric_cols_item] = item_base_features[numeric_cols_item].fillna(0)

    # 3. 商品品类映射
    item_category = df_train.groupby("item_id", observed=True)["category_id"].first().reset_index()
    item_category["item_id"] = item_category["item_id"].astype("int32")
    item_category["category_id"] = item_category["category_id"].astype("int32")

    # 4. 商品新品判断【修正逻辑统一】
    item_life = df_train.groupby("item_id", observed=True).agg(
        item_first_behavior_date=("date", "min"),
        item_behavior_days=("date", "nunique")
    ).reset_index()
    item_life["item_id"] = item_life["item_id"].astype("int32")
    item_life["item_first_behavior_date"] = pd.to_datetime(item_life["item_first_behavior_date"])
    # 新品定义：首次行为在训练集最后2天 + 行为天数≤2
    item_life["item_is_new"] = (
        (item_life["item_first_behavior_date"] >= TRAIN_END_DATE - pd.Timedelta(days=2)) & 
        (item_life["item_behavior_days"] <= 2)
    ).astype("int8")
    
    # 5. 商品爆款判断【修正逻辑缺陷】
    # 仅对有购买的商品计算爆款阈值
    item_with_buy = item_base_features[item_base_features["item_total_buy"] > 0]
    if len(item_with_buy) > 0:
        sales_top10_threshold = item_with_buy["item_total_buy"].quantile(0.9)
    else:
        sales_top10_threshold = 0  # 无购买商品时阈值设为0
    item_base_features["item_is_hot"] = (item_base_features["item_total_buy"] >= sales_top10_threshold).astype("int8")

    # 合并所有商品特征
    item_features = item_base_features.merge(item_category, on="item_id", how="left")
    item_features = item_features.merge(item_life[["item_id", "item_is_new"]], on="item_id", how="left")
    # 数值列填充0
    numeric_cols_item_final = item_features.select_dtypes(include=[np.number]).columns
    item_features[numeric_cols_item_final] = item_features[numeric_cols_item_final].fillna(0)
    
    all_features["item_features"] = item_features
    print(f"商品特征构建完成，共 {item_features.shape[1]-1} 维特征")

    #  4.3 场景特征
    print("正在构建场景特征...")
    def add_scene_features(df):
        df_scene = df.copy()
        # 时段分箱
        bins = [0, 6, 12, 18, 24]
        labels = [4, 1, 2, 3]
        df_scene["time_slot"] = pd.cut(df_scene["hour"], bins=bins, labels=labels, right=False, include_lowest=True).astype("int8")
        # 大促临近天数
        promotion_date = pd.to_datetime("2017-12-12")
        df_scene["date_dt"] = pd.to_datetime(df_scene["date"])
        df_scene["days_to_promotion"] = (promotion_date - df_scene["date_dt"]).dt.days.astype("int8")
        df_scene = df_scene.drop(columns=["date_dt"])
        return df_scene
    
    # 给训练/验证/测试集加场景特征
    df_train_scene = add_scene_features(df_train)
    df_val_scene = add_scene_features(df_val)
    df_test_scene = add_scene_features(df_test)

    # 用户时段偏好特征
    user_time_prefer = df_train_scene.groupby(["user_id", "time_slot"], observed=True).size().reset_index(name="slot_behavior_count")
    user_top_slot = user_time_prefer.sort_values(["user_id", "slot_behavior_count"], ascending=[True, False]).groupby("user_id", observed=True).head(1)
    user_top_slot["user_id"] = user_top_slot["user_id"].astype("int32")
    # time_slot仅4个值，压缩为int8；count保留int32
    user_top_slot["user_prefer_time_slot"] = user_top_slot["time_slot"].astype("int8")
    user_top_slot["user_slot_max_count"] = user_top_slot["slot_behavior_count"].astype("int32")
    user_top_slot = user_top_slot[["user_id", "user_prefer_time_slot", "user_slot_max_count"]]

    all_features["user_time_features"] = user_top_slot
    all_features["df_train_scene"] = df_train_scene
    all_features["df_val_scene"] = df_val_scene
    all_features["df_test_scene"] = df_test_scene
    print(f"场景特征构建完成")

    #  4.4 交叉特征
    print("正在构建交叉特征...")
    # 1. 用户-品类交叉特征
    category_click = df_train[df_train["behavior_type"] == "click"].groupby(
        ["user_id", "category_id"], observed=True
    ).size().reset_index(name="user_category_click_count")
    category_buy = df_train[df_train["behavior_type"] == "buy"].groupby(
        ["user_id", "category_id"], observed=True
    ).size().reset_index(name="user_category_buy_count")
    category_cart = df_train[df_train["behavior_type"] == "cart"].groupby(
        ["user_id", "category_id"], observed=True
    ).size().reset_index(name="user_category_cart_count")
    category_fav = df_train[df_train["behavior_type"] == "fav"].groupby(
        ["user_id", "category_id"], observed=True
    ).size().reset_index(name="user_category_fav_count")
    category_total = df_train.groupby(
        ["user_id", "category_id"], observed=True
    ).size().reset_index(name="user_category_total_behavior")
    
    # 类型压缩
    for df in [category_click, category_buy, category_cart, category_fav, category_total]:
        df[["user_id", "category_id"]] = df[["user_id", "category_id"]].astype("int32")
        df.iloc[:, 2:] = df.iloc[:, 2:].astype("int32")
    
    # 合并交叉特征
    user_category_cross = category_total.merge(category_click, on=["user_id", "category_id"], how="left")
    user_category_cross = user_category_cross.merge(category_buy, on=["user_id", "category_id"], how="left")
    user_category_cross = user_category_cross.merge(category_cart, on=["user_id", "category_id"], how="left")
    user_category_cross = user_category_cross.merge(category_fav, on=["user_id", "category_id"], how="left")
    
    # 仅数值列填充缺失值
    numeric_cols_cross = user_category_cross.select_dtypes(include=[np.number]).columns
    user_category_cross[numeric_cols_cross] = user_category_cross[numeric_cols_cross].fillna(0)
    
    # 交叉特征指标定义
    user_category_cross["user_category_click_ratio"] = (user_category_cross["user_category_click_count"] / user_category_cross["user_category_total_behavior"].replace(0, np.nan)).astype("float32")
    user_category_cross["user_category_cvr"] = (user_category_cross["user_category_buy_count"] / user_category_cross["user_category_click_count"].replace(0, np.nan)).astype("float32")
    user_category_cross["user_category_cart_rate"] = (user_category_cross["user_category_cart_count"] / user_category_cross["user_category_click_count"].replace(0, np.nan)).astype("float32")
    
    # 数值列填充0
    numeric_cols_cross = user_category_cross.select_dtypes(include=[np.number]).columns
    user_category_cross[numeric_cols_cross] = user_category_cross[numeric_cols_cross].fillna(0)

    # 2. 用户-商品交叉特征
    user_item_click = df_train[df_train["behavior_type"] == "click"].groupby(
        ["user_id", "item_id"], observed=True
    ).size().reset_index(name="click_count")  
    user_item_buy = df_train[df_train["behavior_type"] == "buy"].groupby(
        ["user_id", "item_id"], observed=True
    ).size().reset_index(name="buy_count")    
    user_item_total = df_train.groupby(
        ["user_id", "item_id"], observed=True
    ).size().reset_index(name="user_item_interact_count")
    
    # 类型压缩
    for df in [user_item_click, user_item_buy, user_item_total]:
        df[["user_id", "item_id"]] = df[["user_id", "item_id"]].astype("int32")
        df.iloc[:, 2:] = df.iloc[:, 2:].astype("int32")
    
    user_item_interact = user_item_total.merge(user_item_click, on=["user_id", "item_id"], how="left")
    user_item_interact = user_item_interact.merge(user_item_buy, on=["user_id", "item_id"], how="left")
    
    # 仅数值列填充缺失值
    numeric_cols_item_cross = user_item_interact.select_dtypes(include=[np.number]).columns
    user_item_interact[numeric_cols_item_cross] = user_item_interact[numeric_cols_item_cross].fillna(0)
    
    # 后续判断逻辑同步修正
    user_item_interact["user_item_ever_click"] = (user_item_interact["click_count"] > 0).astype("int8")
    user_item_interact["user_item_ever_buy"] = (user_item_interact["buy_count"] > 0).astype("int8")
    user_item_interact = user_item_interact[["user_id", "item_id", "user_item_ever_click", "user_item_ever_buy", "user_item_interact_count"]]

    all_features["user_category_cross"] = user_category_cross
    all_features["user_item_cross"] = user_item_interact
    print(f"交叉特征构建完成")
    print(" 所有特征构建完成 ")
    return all_features

"""特征合并（零报错+内存优化版，完全保留原逻辑）"""
def merge_features_to_sample(df_sample, features):
    """
    合并所有特征到样本，同时构造模型训练标签
    优化：索引join降低内存、类型极致压缩、删除无用列
    零特征穿越、零报错
    """
    df = df_sample.copy()

    # 【核心新增】构造模型训练标签
    # 购买标签：buy=1（正样本），其他行为=0（负样本），用于CVR预估
    df["label"] = df["behavior_type"].apply(lambda x: 1 if x == "buy" else 0).astype("int8")
    # 点击标签：click=1，其他=0，用于CTR预估
    df["click_label"] = df["behavior_type"].apply(lambda x: 1 if x == "click" else 0).astype("int8")

    # 提前删除模型无用列，降低内存
    drop_useless_cols = ["datetime", "date", "timestamp"]
    df = df.drop(columns=[c for c in drop_useless_cols if c in df.columns])

    # 1. 基础ID类型压缩 
    id_cols = ["user_id", "item_id", "category_id"]
    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].astype("int32")

    # 2. 合并用户特征 
    df = df.set_index("user_id")
    user_feat = features["user_features"].set_index("user_id")
    # 删除模型用不上的非数值列
    drop_cols_user = ["user_first_behavior_date"]
    user_feat = user_feat.drop(columns=[c for c in drop_cols_user if c in user_feat.columns])
    # 类型压缩
    for col in user_feat.columns:
        if "float64" == str(user_feat[col].dtype):
            user_feat[col] = user_feat[col].astype("float32")
        elif "int64" == str(user_feat[col].dtype):
            user_feat[col] = user_feat[col].astype("int32")
    df = df.join(user_feat, how="left")
    df = df.reset_index()

    #  3. 合并商品特征 
    df = df.set_index(["item_id", "category_id"])
    item_feat = features["item_features"].set_index(["item_id", "category_id"])
    # 删除无用列
    drop_cols_item = ["date"]
    item_feat = item_feat.drop(columns=[c for c in drop_cols_item if c in item_feat.columns])
    # 类型压缩
    for col in item_feat.columns:
        if "float64" == str(item_feat[col].dtype):
            item_feat[col] = item_feat[col].astype("float32")
        elif "int64" == str(item_feat[col].dtype):
            item_feat[col] = item_feat[col].astype("int32")
    df = df.join(item_feat, how="left")
    df = df.reset_index()

    #  4. 合并用户时段特征 
    df = df.set_index("user_id")
    time_feat = features["user_time_features"].set_index("user_id")
    for col in time_feat.columns:
        if "int64" == str(time_feat[col].dtype):
            time_feat[col] = time_feat[col].astype("int32")
    df = df.join(time_feat, how="left")
    df = df.reset_index()

    # 5. 合并用户-品类交叉特征 
    df = df.set_index(["user_id", "category_id"])
    cate_cross_feat = features["user_category_cross"].set_index(["user_id", "category_id"])
    for col in cate_cross_feat.columns:
        if "float64" == str(cate_cross_feat[col].dtype):
            cate_cross_feat[col] = cate_cross_feat[col].astype("float32")
        elif "int64" == str(cate_cross_feat[col].dtype):
            cate_cross_feat[col] = cate_cross_feat[col].astype("int32")
    df = df.join(cate_cross_feat, how="left")
    df = df.reset_index()

    # 6. 合并用户-商品交叉特征 
    df = df.set_index(["user_id", "item_id"])
    item_cross_feat = features["user_item_cross"].set_index(["user_id", "item_id"])
    for col in item_cross_feat.columns:
        if "int64" == str(item_cross_feat[col].dtype):
            item_cross_feat[col] = item_cross_feat[col].astype("int32")
    df = df.join(item_cross_feat, how="left")
    df = df.reset_index()

    # 7. 填充缺失值 + 最终类型压缩 
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    # 最终类型压缩
    for col in df.columns:
        if "float64" == str(df[col].dtype):
            df[col] = df[col].astype("float32")
        elif "int64" == str(df[col].dtype):
            df[col] = df[col].astype("int32")

    return df

if __name__ == "__main__":
    # 1 数据预处理
    df_clean = data_preprocess(FILE_PATH)

    # 内存不够就打开，抽20%用户跑，内存直接降80%
    # sample_users = df_clean["user_id"].drop_duplicates().sample(frac=0.2, random_state=2024)
    # df_clean = df_clean[df_clean["user_id"].isin(sample_users)].reset_index(drop=True)

    # 2 划分数据集
    df_train, df_val, df_test = split_dataset_by_time_industrial(df_clean)

    # 3 特征工程
    features = build_all_features(df_train, df_val, df_test)

    # = 分块合并+保存，彻底解决内存溢出 =
    print(" 合并特征到样本 ")
    # 分块大小：内存不够就改成 1e6（100万行/块），默认200万行/块
    CHUNK_SIZE = 2_000_000

    #  处理验证集
    # print(" 开始处理验证集 ")
    # val_final = merge_features_to_sample(features["df_val_scene"], features)
    # val_final.to_parquet("./val_final_features.parquet", index=False)
    # print(" 验证集合并保存完成！")

    #  处理训练集
    print(" 开始处理训练集 ")
    train_scene = features["df_train_scene"]
    first_chunk_train = True
    writer_train = None
    try:
        for i in range(0, len(train_scene), CHUNK_SIZE):
            current_block = i//CHUNK_SIZE + 1
            total_block = (len(train_scene) + CHUNK_SIZE -1) // CHUNK_SIZE
            print(f"训练集进度：{current_block}/{total_block} 块，行范围：{i} ~ {min(i+CHUNK_SIZE, len(train_scene))}")
            chunk = train_scene.iloc[i:i+CHUNK_SIZE].copy()
            chunk_final = merge_features_to_sample(chunk, features)
            table = pa.Table.from_pandas(chunk_final)
            if first_chunk_train:
                writer_train = pq.ParquetWriter("./train_final_features.parquet", table.schema)
                first_chunk_train = False
            writer_train.write_table(table)
            del chunk, chunk_final, table
    finally:
        if writer_train:
            writer_train.close()
            print("训练集Parquet写入器已安全关闭")
    print("训练集合并保存完成！")

    #  处理测试集
    print(" 开始处理测试集 ")
    test_scene = features["df_test_scene"]
    
    # 增加空数据判断
    if len(test_scene) == 0:
        print("测试集为空，跳过写入")
    else:
        first_chunk = True
        writer = None
        try:
            for i in range(0, len(test_scene), CHUNK_SIZE):
                current_block = i//CHUNK_SIZE + 1
                total_block = (len(test_scene) + CHUNK_SIZE -1) // CHUNK_SIZE
                print(f"测试集进度：{current_block}/{total_block} 块，行范围：{i} ~ {min(i+CHUNK_SIZE, len(test_scene))}")
                
                chunk = test_scene.iloc[i:i+CHUNK_SIZE].copy()
                chunk_final = merge_features_to_sample(chunk, features)
                
                table = pa.Table.from_pandas(chunk_final)
                if first_chunk:
                    writer = pq.ParquetWriter("./test_final_features.parquet", table.schema)
                    first_chunk = False
                writer.write_table(table)
                
                # 释放临时变量内存
                del chunk, chunk_final, table
        finally:
            if writer:
                writer.close()
                print(" Parquet写入器已安全关闭")
    print(" 测试集合并保存完成！")

    

    # 验证最终结果
    test_check = pd.read_parquet("./test_final_features.parquet")
    print(f"最终测试集特征维度：{test_check.shape}")
    print(f"测试集标签分布：\n{test_check['label'].value_counts()}")
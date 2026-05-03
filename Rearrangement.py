import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import pickle
import gc
import warnings
warnings.filterwarnings('ignore')

#  全局配置：12G显存适配+链路一致性 
try:
    from Rough_ranking import (
        DEVICE, USE_CUDA, DTYPE_CONFIG, BATCH_SIZE
    )
except ImportError:
    # 兜底配置，12G显存专属优化
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = False
    BATCH_SIZE = 4096
    DTYPE_CONFIG = {
        "user_id": np.int32,
        "item_id": np.int32,
        "category_id": np.int32,
        "float_feature": np.float16,
        "int_feature": np.int32,
        "idx_dtype": np.int32
    }

#  重排层核心超参[强制30%+长尾专属]
FINAL_TOP_N = 10                  # 最终输出商品数
MAX_CATEGORY_PER_USER = 6         # [彻底放宽]同品类最多6个，完全不卡长尾的品类限制
MIN_CATEGORY_PER_USER = 2         # [最低要求]仅保2个品类，所有资源向长尾倾斜
MIN_LONGTAIL_PER_USER = 3         # [硬规则，不满足直接不合格]人均必须3个长尾
MAX_BRAND_PER_USER = 5
LONG_TAIL_ALPHA = 100.0           # [拉满]长尾权重100倍，彻底碾压热门商品的CTR优势
DPP_LAMBDA = 1.2
DPP_TRADE_OFF = 0.2               # [几乎不看相关性]彻底给多样性和长尾让路
DPP_LONGTAIL_BOOST = 50.0         # [拉满]DPP里长尾边际收益增益50倍
LONGTAIL_QUANTILE = 0.6           # [扩大长尾池]60%的商品划入长尾，单用户候选长尾直接翻倍
# ========== 结果保存路径 ==========
REARRANGE_SAVE_DIR = r"Saved\5 Rearrange"
REARRANGE_RESULT_SAVE_PATH = os.path.join(REARRANGE_SAVE_DIR, "rearrange_result_full.parquet")
os.makedirs(REARRANGE_SAVE_DIR, exist_ok=True)

#  重排层基类 
class BaseRearrange:
    def __init__(self, top_n=FINAL_TOP_N):
        self.top_n = top_n
        self.name = self.__class__.__name__
        self.save_dir = os.path.join(REARRANGE_SAVE_DIR, self.name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 预计算缓存[全CUDA张量，零字典查询]
        self.item_feature_cache = None
        self.item_id_to_idx = None
        self.item_category_cache = None
        self.item_longtail_weight_cache = None
        self.item_is_longtail_cache = None
        self.click_threshold = 0
        self.is_fitted = False

    def fit(self, item_features_df):
        raise NotImplementedError
    def batch_rearrange(self, user_ids, user_candidate_items, item_scores_list):
        raise NotImplementedError
    def single_rearrange(self, user_id, candidate_items, item_scores):
        raise NotImplementedError
    def save(self):
        raise NotImplementedError
    @classmethod
    def load(cls):
        raise NotImplementedError

#  工业级重排核心实现
class EcommerceRearrange(BaseRearrange):
    def __init__(self, top_n=FINAL_TOP_N):
        super().__init__(top_n)
        self.max_category = MAX_CATEGORY_PER_USER
        self.min_category = MIN_CATEGORY_PER_USER
        self.min_longtail = MIN_LONGTAIL_PER_USER
        self.max_brand = MAX_BRAND_PER_USER
        self.long_tail_alpha = LONG_TAIL_ALPHA
        self.dpp_lambda = DPP_LAMBDA
        self.dpp_trade_off = DPP_TRADE_OFF
        self.dpp_longtail_boost = DPP_LONGTAIL_BOOST
        self.longtail_quantile = LONGTAIL_QUANTILE

    def fit(self, item_features_df):
        """预计算全量商品静态信息，全向量化无循环"""
        print("===== 开始预计算全量商品重排静态信息 =====")
        item_features_df = item_features_df.sort_values("item_id").reset_index(drop=True)
        self.item_id_to_idx = pd.Series(
            item_features_df.index, 
            index=item_features_df["item_id"]
        ).to_dict()
        item_num = len(item_features_df)
        # 1. 品类编码张量
        category_series = item_features_df["category_id"].astype(DTYPE_CONFIG["category_id"])
        self.item_category_cache = torch.IntTensor(category_series.values).to(DEVICE, non_blocking=True)
        # 2. 长尾判定与权重[核心优化：逆点击量加权+分位数阈值]
        self.click_threshold = item_features_df["item_total_click"].quantile(self.longtail_quantile)
        total_click_series = item_features_df["item_total_click"].fillna(0).clip(lower=1)
        max_click = total_click_series.max()
        # [修改]<=阈值，扩大长尾覆盖，避免相同点击量被排除
        is_longtail_np = (total_click_series.values <= self.click_threshold).astype(np.int32)
        self.item_is_longtail_cache = torch.IntTensor(is_longtail_np).to(DEVICE, non_blocking=True)
        # [核心优化]逆点击量长尾加权，彻底抵消热门商品CTR优势
        longtail_weight_np = 1.0 + self.long_tail_alpha * np.power(max_click / total_click_series.values, 0.6)
        self.item_longtail_weight_cache = torch.FloatTensor(longtail_weight_np).to(DEVICE, non_blocking=True)
        # 3. DPP特征张量
        feat_cols = ["item_ctr", "item_cvr", "item_cart_rate", "item_fav_rate", "item_is_hot", "item_is_new"]
        feat_np = item_features_df[feat_cols].fillna(0).values.astype(np.float32)
        feat_np = (feat_np - feat_np.mean(axis=0)) / (feat_np.std(axis=0) + 1e-8)
        self.item_feature_cache = torch.FloatTensor(feat_np).to(DEVICE, non_blocking=True)

        #  预计算全量长尾商品池，用于终极兜底 
        self.all_longtail_item_ids = item_features_df[is_longtail_np == 1]["item_id"].tolist()
        self._has_printed_longtail_avg = False  # 重置打印标记，避免运行异常
        # ======

        self.is_fitted = True
        print(f" 预计算完成，共缓存 {item_num} 个商品，长尾商品占比：{is_longtail_np.mean():.2%}，已全部加载至{DEVICE}")
        print(f" 预计算全量长尾兜底池，共 {len(self.all_longtail_item_ids)} 个长尾商品")
        self.save()
        return self
    
    def _batch_build_candidate_info(self, user_candidate_items, item_scores_list):
        """预分配内存+批量处理，消除循环冗余，GPU异步拷贝提速"""
        user_candidate_len = [len(items) for items in user_candidate_items]
        max_candidate_len = max(user_candidate_len)
        user_num = len(user_candidate_items)

        # 预分配全量数组，一次性填充，减少循环内内存分配
        candidate_idx_np = np.zeros((user_num, max_candidate_len), dtype=DTYPE_CONFIG["idx_dtype"])
        candidate_scores_np = np.zeros((user_num, max_candidate_len), dtype=np.float32)
        padding_mask_np = np.zeros((user_num, max_candidate_len), dtype=bool)

        # 预提取item_id_to_idx的keys，加速in判断
        item_id_set = set(self.item_id_to_idx.keys())
        # 循环内仅做最核心的赋值，无冗余操作
        for user_idx in range(user_num):
            items = user_candidate_items[user_idx]
            scores = item_scores_list[user_idx]
            valid_len = len(items)
            if valid_len == 0:
                continue
            # 向量化过滤有效item
            valid_mask = np.array([item in item_id_set for item in items], dtype=bool)
            valid_items = np.array(items)[valid_mask]
            valid_scores = np.array(scores)[valid_mask]
            valid_count = len(valid_items)
            if valid_count == 0:
                continue
            # 一次性填充数组
            candidate_idx_np[user_idx, :valid_count] = [self.item_id_to_idx[item] for item in valid_items]
            candidate_scores_np[user_idx, :valid_count] = valid_scores
            padding_mask_np[user_idx, :valid_count] = True

        # 异步GPU拷贝，非阻塞执行，CPU和GPU并行
        candidate_idx_tensor = torch.IntTensor(candidate_idx_np).to(DEVICE, non_blocking=True)
        candidate_scores_tensor = torch.FloatTensor(candidate_scores_np).to(DEVICE, non_blocking=True)
        padding_mask_tensor = torch.BoolTensor(padding_mask_np).to(DEVICE, non_blocking=True)

        # 批量索引获取属性，GPU全并行，无开销
        batch_category = self.item_category_cache[candidate_idx_tensor]
        batch_longtail_weight = self.item_longtail_weight_cache[candidate_idx_tensor]
        batch_is_longtail = self.item_is_longtail_cache[candidate_idx_tensor]
        batch_item_feat = self.item_feature_cache[candidate_idx_tensor]

        # 合并归一化+加权计算，减少中间张量创建
        log_scores = torch.log1p(candidate_scores_tensor * 1000)
        user_min_score = log_scores.min(dim=1, keepdim=True)[0]
        user_max_score = log_scores.max(dim=1, keepdim=True)[0]
        norm_scores = (log_scores - user_min_score) / (user_max_score - user_min_score + 1e-8)
        norm_scores = torch.pow(norm_scores, 0.2)
        weighted_scores = norm_scores * batch_longtail_weight
        weighted_scores[~padding_mask_tensor] = -torch.inf

        # 仅第一块打印，避免IO阻塞
        if user_num > 1000 and not hasattr(self, '_has_printed_longtail_avg'):
            avg_longtail_per_user = batch_is_longtail.sum(dim=1).float().mean().item()
            print(f" 单用户候选集平均长尾商品数：{avg_longtail_per_user:.2f} 个")
            self._has_printed_longtail_avg = True

        # 清理大张量，仅函数结束一次
        del candidate_scores_tensor, norm_scores, user_min_score, user_max_score, log_scores

        return (
            candidate_idx_tensor, weighted_scores, padding_mask_tensor,
            batch_category, batch_is_longtail, batch_item_feat, user_candidate_len
        )
        
    def _batch_dpp_rerank(self, weighted_scores, item_feat_tensor, padding_mask, batch_category, batch_is_longtail):
        """[强制长尾锁死版]先100%选满3个长尾，完全不卡品类，再处理剩余坑位"""
        user_num, max_len = weighted_scores.shape
        final_selected_idx = torch.zeros((user_num, self.top_n), device=DEVICE, dtype=torch.int64)
        final_selected_mask = torch.zeros((user_num, self.top_n), device=DEVICE, dtype=torch.bool)

        valid_user_mask = torch.any(padding_mask, dim=1)
        valid_user_num = valid_user_mask.sum().item()
        if valid_user_num == 0:
            return final_selected_idx, final_selected_mask

        # 提取有效用户张量
        valid_weighted_scores = weighted_scores[valid_user_mask]
        valid_item_feat = item_feat_tensor[valid_user_mask]
        valid_padding_mask = padding_mask[valid_user_mask]
        valid_category = batch_category[valid_user_mask]
        valid_is_longtail = batch_is_longtail[valid_user_mask]
        valid_user_original_idx = torch.where(valid_user_mask)[0]
        valid_user_num, max_len, feat_dim = valid_item_feat.shape

        # 预计算固定张量
        batch_unique_cats, category_idx_tensor = torch.unique(valid_category, return_inverse=True)
        batch_category_num = len(batch_unique_cats)
        selected_category_count = torch.zeros((valid_user_num, batch_category_num), device=DEVICE, dtype=torch.int32)
        feat_norm = torch.nn.functional.normalize(valid_item_feat, dim=-1)
        selected_mask = torch.zeros((valid_user_num, max_len), device=DEVICE, dtype=torch.bool)
        selected_indices = torch.full((valid_user_num, self.top_n), fill_value=-1, device=DEVICE, dtype=torch.int64)

        #  [硬规则：先锁死3个长尾，完全不卡品类，不卡任何限制]
        longtail_scores = valid_weighted_scores.clone()
        longtail_scores[valid_is_longtail != 1] = -torch.inf
        longtail_scores[~valid_padding_mask] = -torch.inf

        for longtail_step in range(self.min_longtail):
            # [完全放开限制]只屏蔽已选商品，其他任何限制都不加，先选满3个长尾再说
            step_scores = longtail_scores.clone()
            step_scores[selected_mask] = -torch.inf

            best_idx = torch.argmax(step_scores, dim=1)
            step_valid_mask = step_scores[torch.arange(valid_user_num), best_idx] > -torch.inf

            # 仅统计品类，不做任何限制
            selected_cat_idx = category_idx_tensor[torch.arange(valid_user_num), best_idx]
            selected_category_count.scatter_add_(
                1, selected_cat_idx.unsqueeze(1),
                torch.ones_like(selected_cat_idx.unsqueeze(1), dtype=torch.int32)
            )

            # 锁死长尾坑位，绝对不允许被替换
            selected_indices[step_valid_mask, longtail_step] = best_idx[step_valid_mask]
            selected_mask[step_valid_mask, best_idx[step_valid_mask]] = True
            longtail_scores[step_valid_mask, best_idx[step_valid_mask]] = -torch.inf

        #  剩余7个坑位：优先选长尾，再选热门，品类仅做软限制 
        remain_step_start = self.min_longtail
        for step in range(remain_step_start, self.top_n):
            # 剩余坑位仅做软限制，超过品类上限仅降权，不屏蔽
            current_category_count = torch.gather(selected_category_count, dim=1, index=category_idx_tensor)
            over_limit_penalty = (current_category_count >= self.max_category).float() * 1000

            # 计算DPP边际收益，长尾继续拉满增益
            if step == remain_step_start:
                marginal_gain = valid_weighted_scores ** 2
            else:
                prev_selected_idx = selected_indices[:, :step]
                prev_selected_idx[prev_selected_idx == -1] = 0
                selected_feat = torch.gather(feat_norm, dim=1, index=prev_selected_idx.unsqueeze(-1).expand(-1, -1, feat_dim))
                sim_square_sum = torch.sum(torch.bmm(selected_feat, feat_norm.permute(0, 2, 1)) ** 2, dim=1)
                marginal_gain = (1 - sim_square_sum) * (valid_weighted_scores ** 2)

            # 长尾增益拉满，超品类仅降权不屏蔽
            marginal_gain[valid_is_longtail == 1] *= self.dpp_longtail_boost
            marginal_gain = marginal_gain - over_limit_penalty
            # 仅屏蔽已选和无效商品
            marginal_gain[selected_mask | ~valid_padding_mask] = -torch.inf

            # 选最优商品
            best_idx = torch.argmax(marginal_gain, dim=1)
            step_valid_mask = marginal_gain[torch.arange(valid_user_num), best_idx] > -torch.inf

            # 更新品类计数
            selected_cat_idx = category_idx_tensor[torch.arange(valid_user_num), best_idx]
            selected_category_count.scatter_add_(
                1, selected_cat_idx.unsqueeze(1),
                torch.ones_like(selected_cat_idx.unsqueeze(1), dtype=torch.int32)
            )

            # 保存结果
            selected_indices[step_valid_mask, step] = best_idx[step_valid_mask]
            selected_mask[step_valid_mask, best_idx[step_valid_mask]] = True

        # 映射回原batch
        final_selected_idx[valid_user_original_idx] = selected_indices
        final_selected_mask[valid_user_original_idx] = selected_indices != -1

        # 清理显存
        del feat_norm, valid_weighted_scores, valid_item_feat, longtail_scores
        return final_selected_idx, final_selected_mask
    
    def batch_rearrange(self, user_ids, user_candidate_items, item_scores_list):
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=USE_CUDA):
            if not self.is_fitted:
                raise RuntimeError("请先调用fit方法预计算商品信息")
            user_num = len(user_ids)
            if user_num == 0:
                return []

            # 步骤1：构建候选信息
            (
                candidate_idx_tensor, weighted_scores, padding_mask_tensor,
                batch_category, batch_is_longtail, batch_item_feat, user_candidate_len
            ) = self._batch_build_candidate_info(user_candidate_items, item_scores_list)

            # 步骤2：DPP重排，先锁死3个长尾
            selected_idx_tensor, selected_mask_tensor = self._batch_dpp_rerank(
                weighted_scores, batch_item_feat, padding_mask_tensor, batch_category, batch_is_longtail
            )

            # 核心提速：一次性全量拷贝到CPU，后续全numpy向量化，无单用户GPU交互
            weighted_scores_cpu = weighted_scores.cpu().numpy()
            batch_is_longtail_cpu = batch_is_longtail.cpu().numpy()
            selected_idx_tensor_cpu = selected_idx_tensor.cpu().numpy()
            selected_mask_tensor_cpu = selected_mask_tensor.cpu().numpy()
            item_is_longtail_cache_cpu = self.item_is_longtail_cache.cpu().numpy()
            item_id_to_idx_np = np.array(list(self.item_id_to_idx.items()), dtype=object)
            item_id_arr = item_id_to_idx_np[:, 0].astype(np.int32)
            item_idx_arr = item_id_to_idx_np[:, 1].astype(np.int32)
            item_id_to_idx_map = np.vectorize(lambda x: self.item_id_to_idx.get(x, -1))

            # 步骤3：全量结果批量生成，最小化Python循环
            final_result = []
            # 预先生成全量长尾兜底池的numpy数组，避免循环内重复访问
            all_longtail_np = np.array(self.all_longtail_item_ids, dtype=np.int32)

            for user_idx in range(user_num):
                current_candidate_items = user_candidate_items[user_idx]
                candidate_len = user_candidate_len[user_idx]
                if candidate_len == 0:
                    final_result.append(all_longtail_np[:self.top_n].tolist())
                    continue

                # 提取DPP选中的商品，向量化去重
                valid_pos = selected_idx_tensor_cpu[user_idx, selected_mask_tensor_cpu[user_idx]]
                valid_pos = valid_pos[valid_pos < candidate_len]
                valid_pos = list(dict.fromkeys(valid_pos))
                valid_items = [current_candidate_items[pos] for pos in valid_pos]
                valid_items = valid_items[:self.top_n]

                #  [100%保留原逻辑]长尾统计，向量化加速 
                valid_items_np = np.array(valid_items, dtype=np.int32)
                valid_item_idx = item_id_to_idx_map(valid_items_np)
                valid_mask = valid_item_idx != -1
                valid_longtail_mask = item_is_longtail_cache_cpu[valid_item_idx[valid_mask]] == 1

                # 拆分长尾/非长尾
                selected_longtail = valid_items_np[valid_mask][valid_longtail_mask].tolist()
                selected_non_longtail = valid_items_np[~valid_mask].tolist() + valid_items_np[valid_mask][~valid_longtail_mask].tolist()
                need_longtail_num = max(0, self.min_longtail - len(selected_longtail))

                #  候选池补长尾，向量化加速 
                if need_longtail_num > 0:
                    # 向量化找候选池里的未选长尾
                    candidate_items_np = np.array(current_candidate_items, dtype=np.int32)
                    candidate_idx = item_id_to_idx_map(candidate_items_np)
                    candidate_valid_mask = candidate_idx != -1
                    candidate_longtail_mask = item_is_longtail_cache_cpu[candidate_idx[candidate_valid_mask]] == 1
                    candidate_longtail_items = candidate_items_np[candidate_valid_mask][candidate_longtail_mask]
                    # 过滤已选商品
                    candidate_longtail_items = candidate_longtail_items[~np.isin(candidate_longtail_items, valid_items_np)]
                    # 按分数排序
                    candidate_longtail_scores = weighted_scores_cpu[user_idx][np.isin(candidate_items_np, candidate_longtail_items)]
                    sorted_idx = np.argsort(-candidate_longtail_scores)
                    candidate_longtail_sorted = candidate_longtail_items[sorted_idx]

                    # 补全长尾
                    add_longtail = candidate_longtail_sorted[:need_longtail_num]
                    selected_longtail += add_longtail.tolist()
                    selected_non_longtail = selected_non_longtail[:max(0, len(selected_non_longtail) - len(add_longtail))]
                    need_longtail_num = max(0, self.min_longtail - len(selected_longtail))

                #  全局兜底补长尾，硬规则保证3个 
                if need_longtail_num > 0:
                    global_add = all_longtail_np[:need_longtail_num]
                    selected_longtail += global_add.tolist()
                    selected_non_longtail = selected_non_longtail[:max(0, len(selected_non_longtail) - need_longtail_num)]

                # 重新构建结果，长尾在前保证不被挤掉
                valid_items = selected_longtail + selected_non_longtail

                #  补全空位到10个，向量化加速 
                need_fill_num = self.top_n - len(valid_items)
                if need_fill_num > 0:
                    # 向量化找剩余候选
                    candidate_items_np = np.array(current_candidate_items, dtype=np.int32)
                    unselected_mask = ~np.isin(candidate_items_np, valid_items)
                    unselected_items = candidate_items_np[unselected_mask]
                    unselected_scores = weighted_scores_cpu[user_idx][unselected_mask]
                    unselected_idx = item_id_to_idx_map(unselected_items)
                    unselected_valid_mask = unselected_idx != -1

                    # 拆分长尾/热门，优先补长尾
                    unselected_longtail_mask = item_is_longtail_cache_cpu[unselected_idx[unselected_valid_mask]] == 1
                    unselected_longtail = unselected_items[unselected_valid_mask][unselected_longtail_mask]
                    unselected_hot = unselected_items[unselected_valid_mask][~unselected_longtail_mask]
                    # 按分数排序
                    unselected_longtail = unselected_longtail[np.argsort(-unselected_scores[unselected_valid_mask][unselected_longtail_mask])]
                    unselected_hot = unselected_hot[np.argsort(-unselected_scores[unselected_valid_mask][~unselected_longtail_mask])]

                    # 补全
                    fill_items = np.concatenate([unselected_longtail, unselected_hot, all_longtail_np])
                    valid_items += fill_items[:need_fill_num].tolist()

                # 最终去重+截断
                valid_items = list(dict.fromkeys(valid_items))[:self.top_n]
                final_result.append(valid_items)

            # [仅函数结束清理一次显存，避免循环内频繁操作]
            del candidate_idx_tensor, weighted_scores, padding_mask_tensor
            del batch_category, batch_is_longtail, batch_item_feat
            del selected_idx_tensor, selected_mask_tensor
            torch.cuda.empty_cache()

            return final_result
    
    def single_rearrange(self, user_id, candidate_items, item_scores):
        final_result = self.batch_rearrange([user_id], [candidate_items], [item_scores])
        return final_result[0]

    def save(self):
        config_dict = {
            "top_n": self.top_n,
            "max_category": self.max_category,
            "min_category": self.min_category,
            "min_longtail": self.min_longtail,
            "max_brand": self.max_brand,
            "long_tail_alpha": self.long_tail_alpha,
            "dpp_lambda": self.dpp_lambda,
            "dpp_trade_off": self.dpp_trade_off,
            "dpp_longtail_boost": self.dpp_longtail_boost,
            "longtail_quantile": self.longtail_quantile,
            "item_id_to_idx": self.item_id_to_idx,
            "click_threshold": self.click_threshold,
            # 保存全量长尾池长度，用于加载校验
            "all_longtail_item_num": len(self.all_longtail_item_ids)
        }
        with open(os.path.join(self.save_dir, "rearrange_config.pkl"), "wb") as f:
            pickle.dump(config_dict, f)
        
        # 单独保存全量长尾商品池
        torch.save(torch.IntTensor(self.all_longtail_item_ids).cpu(), os.path.join(self.save_dir, "all_longtail_item_ids.pt"))
        # 原有缓存保存不变
        torch.save(self.item_category_cache.cpu(), os.path.join(self.save_dir, "item_category_cache.pt"))
        torch.save(self.item_longtail_weight_cache.cpu(), os.path.join(self.save_dir, "item_longtail_weight_cache.pt"))
        torch.save(self.item_is_longtail_cache.cpu(), os.path.join(self.save_dir, "item_is_longtail_cache.pt"))
        torch.save(self.item_feature_cache.cpu(), os.path.join(self.save_dir, "item_feature_cache.pt"))
        print(f"[{self.name}]模型与缓存保存完成！包含全量长尾兜底池")
    
    
    @classmethod
    def load(cls):
        model = cls()
        save_dir = model.save_dir
        
        with open(os.path.join(save_dir, "rearrange_config.pkl"), "rb") as f:
            config_dict = pickle.load(f)
        # 原有参数加载不变
        model.top_n = config_dict["top_n"]
        model.max_category = config_dict["max_category"]
        model.min_category = config_dict["min_category"]
        model.min_longtail = config_dict["min_longtail"]
        model.max_brand = config_dict["max_brand"]
        model.long_tail_alpha = config_dict["long_tail_alpha"]
        model.dpp_lambda = config_dict["dpp_lambda"]
        model.dpp_trade_off = config_dict["dpp_trade_off"]
        model.dpp_longtail_boost = config_dict["dpp_longtail_boost"]
        model.longtail_quantile = config_dict["longtail_quantile"]
        model.item_id_to_idx = config_dict["item_id_to_idx"]
        model.click_threshold = config_dict["click_threshold"]
        
        # 加载全量长尾商品池，彻底修复属性缺失
        model.all_longtail_item_ids = torch.load(os.path.join(save_dir, "all_longtail_item_ids.pt")).cpu().numpy().tolist()
        model._has_printed_longtail_avg = False  # 重置打印标记，避免运行异常
        # 原有缓存加载不变
        model.item_category_cache = torch.load(os.path.join(save_dir, "item_category_cache.pt")).to(DEVICE)
        model.item_longtail_weight_cache = torch.load(os.path.join(save_dir, "item_longtail_weight_cache.pt")).to(DEVICE)
        model.item_is_longtail_cache = torch.load(os.path.join(save_dir, "item_is_longtail_cache.pt")).to(DEVICE)
        model.item_feature_cache = torch.load(os.path.join(save_dir, "item_feature_cache.pt")).to(DEVICE)
        
        model.is_fitted = True
        print(f"[{model.name}]模型加载完成！已加载全量长尾兜底池，共 {len(model.all_longtail_item_ids)} 个长尾商品")
        return model
    
#  模型工厂 
REARRANGE_MODEL_MAP = {
    "EcommerceRearrange": EcommerceRearrange
}

def get_rearrange_model(model_name):
    if model_name not in REARRANGE_MODEL_MAP:
        raise ValueError(f"模型{model_name}不存在，支持的模型：{list(REARRANGE_MODEL_MAP.keys())}")
    return REARRANGE_MODEL_MAP[model_name]()
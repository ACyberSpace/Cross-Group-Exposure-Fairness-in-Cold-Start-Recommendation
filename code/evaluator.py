import pickle
import numpy as np

import itertools
import sys
import heapq
import torch
import world

def argmax_top_k(a, top_k=50):
    ele_idx = heapq.nlargest(top_k, zip(a, itertools.count()))
    return np.array([idx for ele, idx in ele_idx], dtype=np.intc)


def recall(rank, ground_truth, item_group, i_group_num, top_k):
    result = np.zeros((top_k, i_group_num), dtype=np.float32)
    for idx, item in enumerate(rank):
        if item in ground_truth:
            last_idx = idx
            i = ground_truth.index(item)
            result[last_idx:, item_group[i]] += 1.0
    return result


def eval_intersection_result(score_matrix, test_items, top_k=50, batch_item_group=None, i_group_num=0,
                             pre_recall_mat=None, user_batch=None):
    batch_result_recall = []
    for idx in range(len(test_items)):
        scores = score_matrix[idx]
        test_item = test_items[idx]
        item_group = batch_item_group[idx]
        user_id = user_batch[idx]

        ranking = argmax_top_k(scores, top_k)

        recall_row = recall(ranking, test_item, item_group, i_group_num, top_k)
        pre_recall_mat_copy = np.copy(pre_recall_mat[user_id])
        pre_recall_mat_copy = np.where(pre_recall_mat_copy > 0, pre_recall_mat_copy, 1e-6)
        recall_row = recall_row / pre_recall_mat_copy

        batch_result_recall.append(recall_row)

    return batch_result_recall


class Fairness_Evaluator(object):
    def __init__(self, path, dataloader, dname):
        super(Fairness_Evaluator, self).__init__()

        user_group = path + '/user_group.pkl'
        item_group = path + '/item_group.pkl'

        self.user_group = dataloader.user_group
        self.item_group = dataloader.item_group
        self.u_cold = dataloader.u_cold
        self.i_cold = dataloader.genre2idx[dataloader.i_cold]
        self.loader = dataloader

        if dname == "ml-1m":
            self.u_group_num = 7
            self.i_group_num = 9
            self.u_filter = [0, 1, 2, 3, 4, 5, 6]
            self.i_filter = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        if dname == "ml-100k":
            self.u_group_num = 7
            self.i_group_num = 9
            self.u_filter = [0, 1, 2, 3, 4, 5, 6]
            self.i_filter = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        elif dname == "LFM2b":
            self.u_group_num = 2
            self.i_group_num = 4
            self.u_filter = [0, 1]
            self.i_filter = [0, 1, 2, 3]

        self.n_users, self.n_items = dataloader.n_user, dataloader.m_item

        self.pi_items = self.loader.genre_ratio
        self.pi_users = self.loader.attr_ratio

    def load_data(self):
        inter_num = np.zeros((self.u_group_num, self.i_group_num))
        train_user = self.loader.trainUser
        train_item = self.loader.trainItem
        valid_user = self.loader.validUser
        valid_item = self.loader.validItem
        test_user = self.loader.testUser
        test_item = self.loader.testItem
        for u, i in zip(train_user, train_item):
            inter_num[self.user_group[u], self.item_group[i]] += 1
        for u, i in zip(valid_user, valid_item):
            inter_num[self.user_group[u], self.item_group[i]] += 1
        for u, i in zip(test_user, test_item):
            inter_num[self.user_group[u], self.item_group[i]] += 1

        self.pre_recall_mat = np.zeros((self.n_users, self.i_group_num))
        # for u in range(self.n_users):
        #     for i in self.test_set[u]:
        #         self.pre_recall_mat[u][self.item_group[i]] += 1.0

        for u, i in zip(test_user, test_item):
            self.pre_recall_mat[u][self.item_group[i]] += 1.0

        self.user_group_count = {}  # calculate interested users U(i,j)
        for k, v in self.user_group.items():
            if v not in self.user_group_count:
                self.user_group_count[v] = np.zeros(len(self.i_filter))
            for i in range(len(self.i_filter)):
                if self.pre_recall_mat[k][i] > 0:
                    self.user_group_count[v][i] += 1

    def evaluate(self, model):
        results = {}
        users_to_test = list(self.loader.testDict.keys())

        top_show = [10, 20]
        max_top = max(top_show)

        u_batch_size = 1024

        test_users = users_to_test
        n_test_users = len(test_users)
        n_user_batchs = n_test_users // u_batch_size + 1

        intersection_recall_matrix = np.zeros((len(top_show), self.u_group_num, self.i_group_num))

        item_batch = range(self.n_items)

        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size

            user_batch = test_users[start: end]
            batch_users_gpu = torch.Tensor(user_batch).long()
            batch_users_gpu = batch_users_gpu.to(world.device)
            batch_user_group = []
            for u_id in user_batch:
                batch_user_group.append(self.user_group[u_id])

            rate_batch = model.getUsersRating(batch_users_gpu)
            # rate_batch = model.batch_ratings
            rate_batch = np.array(rate_batch.detach().cpu())

            test_items = []
            batch_item_group = []

            for user in user_batch:
                test_items.append(self.loader.testDict[user])

                temp = []
                for item in test_items[-1]:
                    temp.append(self.item_group[item])
                batch_item_group.append(temp)

            for idx, user in enumerate(user_batch):
                train_items_off = self.loader.allPos[user]
                rate_batch[idx][train_items_off] = -np.inf
                # valid_items_off = self.valid_set[user]
                # rate_batch[idx][valid_items_off] = -np.inf

            _, rating_K = torch.topk(torch.tensor(rate_batch), k=30)
            batch_recall_result = eval_intersection_result(rate_batch, test_items, max_top, batch_item_group,
                                                           self.i_group_num, self.pre_recall_mat,
                                                           user_batch)  # [user, topk, item_group]


            for u_idx, user in enumerate(batch_user_group):
                for i, top in enumerate(top_show):
                    intersection_recall_matrix[i, user] += batch_recall_result[u_idx][top - 1]

        for u_key in self.u_filter:
            intersection_recall_matrix[:, u_key] = intersection_recall_matrix[:, u_key] / self.user_group_count[u_key]

        recall_list = []
        for i in range(len(top_show)):
            l = []
            for u_key in self.u_filter:
                # l += [j for j in intersection_recall_matrix[i][u_key] if j > 0]
                l += [j for j in intersection_recall_matrix[i][u_key]]
            recall_list.append(l)

        assert (len(recall_list[1]) == self.u_group_num * self.i_group_num)
        results['recall_matrix_var'] = [np.std(l) / np.mean(l) for l in recall_list]
        results['recall_matrix_min_25'] = [np.mean(np.sort(l)[:len(l) // 4]) for l in recall_list]

        def cal_u_side(recall_list):
            u_side = []
            i_cold_side = []
            for l in recall_list:
                mat = np.array(l).reshape(self.u_group_num, self.i_group_num)
                u_cov = 0
                i_cold_cov = 0
                for i in range(self.i_group_num):
                    u_cov += np.std(mat[:, i]) / (np.mean(mat[:, i]) + 1e-6)
                    if i == self.i_cold:
                        i_cold_cov += np.std(mat[:, i]) / (np.mean(mat[:, i]) + 1e-6)
                u_cov = u_cov / self.i_group_num
                u_side.append(u_cov)
                i_cold_side.append(i_cold_cov)
            return u_side, i_cold_side

        def cal_i_side(recall_list):
            i_side = []
            u_cold_side = []
            for l in recall_list:
                mat = np.array(l).reshape(self.u_group_num, self.i_group_num)
                i_cov = 0
                u_cold_cov = 0
                for i in range(self.u_group_num):
                    i_cov += np.std(mat[i, :]) / (np.mean(mat[i, :])+1e-6)
                    if i == self.u_cold:
                        u_cold_cov += np.std(mat[i, :]) / (np.mean(mat[i, :])+1e-6)
                i_cov = i_cov / self.u_group_num
                i_side.append(i_cov)
                u_cold_side.append(u_cold_cov)
            return i_side, u_cold_side
        u_side_diff, i_cold_diff = cal_u_side(recall_list=recall_list)
        i_side_diff, u_cold_diff = cal_i_side(recall_list=recall_list)

        self.performance_list = recall_list
        results['recall_matrix_uside'] = u_side_diff
        results['recall_matrix_iside'] = i_side_diff
        results['recall_matrix_u_cold_side'] = u_cold_diff
        results['recall_matrix_i_cold_side'] = i_cold_diff


        return results

    def evaluate_exposure(self, model):
        """
        只返回两端曝光度差异（Gini后取平均）:
          - I_side_avg: 按行(用户组) E[u,:]/pi_items -> Gini -> 对行取平均
          - U_side_avg: 按列(物品组) E[:,i]/pi_users -> Gini -> 对列取平均
        """
        results = {}
        top_show = [10, 20]
        max_top = max(top_show)
        u_batch_size = 1024
        eps = 1e-8

        # 先验分布（你已在 loader 中提供）
        pi_items = np.asarray(self.pi_items, dtype=np.float32)  # (i_group_num,)
        pi_users = np.asarray(self.pi_users, dtype=np.float32)  # (u_group_num,)
        pi_items = np.where(pi_items > 0, pi_items, eps)
        pi_users = np.where(pi_users > 0, pi_users, eps)

        def gini(x: np.ndarray) -> float:
            x = np.asarray(x, dtype=np.float64)
            if x.size == 0:
                return 0.0
            m = x.mean()
            if m <= 0:
                return 0.0
            n = x.size
            return float(np.abs(x[:, None] - x[None, :]).sum() / (2.0 * n * n * m))

        # item -> group 映射（向量化）
        item2group = np.fromiter((self.item_group[i] for i in range(self.n_items)),
                                 dtype=np.int64, count=self.n_items)

        # 为每个@k维护一个 原始曝光计数矩阵 (不平均、不归一)
        E_counts = [np.zeros((self.u_group_num, self.i_group_num), dtype=np.float32)
                    for _ in range(len(top_show))]

        users_to_test = list(self.loader.testDict.keys())
        n_test_users = len(users_to_test)
        n_user_batchs = n_test_users // u_batch_size + 1

        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size
            user_batch = users_to_test[start:end]
            if not user_batch:
                continue

            batch_users_gpu = torch.tensor(user_batch, dtype=torch.long, device=world.device)
            scores_t = model.getUsersRating(batch_users_gpu).detach()  # (B, n_items)
            scores = scores_t.cpu().numpy()

            # 屏蔽训练集物品
            for bi, u in enumerate(user_batch):
                scores[bi][self.loader.allPos[u]] = -np.inf

            k_eff = min(max_top, scores.shape[1])
            # top-k 索引（numpy实现）
            top_idx = np.argpartition(-scores, kth=k_eff - 1, axis=1)[:, :k_eff]
            row_idx = np.arange(top_idx.shape[0])[:, None]
            top_idx = top_idx[row_idx, np.argsort(-scores[row_idx, top_idx], axis=1)]

            # 累计计数到 E_counts
            for bi, u in enumerate(user_batch):
                u_g = self.user_group[u]
                gseq = item2group[top_idx[bi]]
                for t_i, k in enumerate(top_show):
                    use_k = min(k, k_eff)
                    if use_k <= 0:
                        continue
                    cnt = np.bincount(gseq[:use_k], minlength=self.i_group_num).astype(np.float32)
                    E_counts[t_i][u_g] += cnt

        # 计算两端差异（只保留平均后的两个量）
        I_side_avg, U_side_avg = [], []
        I_cold_side, U_cold_side = [], []
        Quaility_weighted_exposure = []
        for t_i in range(len(top_show)):
            E = E_counts[t_i]  # (u_group_num, i_group_num)
            P = self.performance_list[t_i]
            P_m = np.array(P).reshape(self.u_group_num, self.i_group_num)
            Q_E = E / (P_m+1e-6)
            Q_E_gap = np.std(Q_E)/np.mean(Q_E)
            Quaility_weighted_exposure.append(Q_E_gap)

            # I端：每行除以 pi_items -> 行Gini -> 平均
            row_ginis = [gini(E[u, :] / pi_items) for u in range(self.u_group_num)]
            I_side_avg.append(float(np.mean(row_ginis)) if row_ginis else 0.0)
            U_cold_side.append(gini(E[self.u_cold, :] / pi_items))

            # U端：每列除以 pi_users -> 列Gini -> 平均
            col_ginis = [gini(E[:, i] / pi_users) for i in range(self.i_group_num)]
            U_side_avg.append(float(np.mean(col_ginis)) if col_ginis else 0.0)
            I_cold_side.append(gini(E[:, self.i_cold] / pi_users))

        results['exposure_matrix_uside'] = U_side_avg
        results['exposure_matrix_iside'] = I_side_avg
        results['exposure_matrix_u_cold_side'] = U_cold_side
        results['exposure_matrix_i_cold_side'] = I_cold_side

        return results
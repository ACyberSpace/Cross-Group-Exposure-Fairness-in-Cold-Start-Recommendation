"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import copy
import heapq
import os
import random
from collections import defaultdict
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from scipy.sparse.linalg import gmres
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix, coo_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time
from FairReweight import FairGATReweighter

EPS = 1e-6

class BasicDataset(Dataset):
    def __init__(self):
        self.itemTail = None
        self.userTail = None
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    @property
    def user_text_feature(self):
        raise NotImplementedError
    @property
    def item_text_feature(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError
    def getSemanticGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A =
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self,config = world.config,path="../data/gowalla"):
        # train or test
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        # self.folds = config['A_n_fold']
        # self.mode_dict = {'train': 0, "test": 1}
        # self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        valid_file = path + '/valid.txt'
        test_file = path + '/test.txt'
        user_path = os.path.join(path, 'users.dat')
        item_path = os.path.join(path, 'items.dat')
        self.path = path
        self.user_info = pd.read_csv(user_path, header=None,
                                     names=["UserID", "Gender", "Age", "Occupation", "Inter_num"], sep='\t')
        self.item_info = pd.read_csv(item_path, header=None,
                                     names=["ItemID", "Title", "Year", "Genres", "Inter_num"], sep='\t', engine='python')
        self.inter_info = pd.read_csv(train_file, header=None, names=["UserID", "ItemID", "Rating"], sep='\t')
        self._user_text_feature = torch.load(self.path + '/user_embedding_tensor.pt').to(world.device)
        self._item_text_feature = torch.load(self.path + '/item_embedding_tensor.pt').to(world.device)
        trainUniqueUsers, trainItem, trainUser, trainRating = [], [], [], []
        validUniqueUsers, validItem, validUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0
        self.validDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split('\t')
                    items = int(l[1])
                    uid = int(l[0])
                    rate = float(l[2])
                    if uid not in trainUniqueUsers:
                        trainUniqueUsers.append(uid)
                    trainUser.append(uid)
                    trainItem.append(items)
                    trainRating.append(rate)
                    self.m_item = max(self.m_item, items)
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += 1
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        self.trainRating = np.array(trainRating)

        with open(valid_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split('\t')
                    items = int(l[1])
                    uid = int(l[0])
                    if uid not in validUniqueUsers:
                        validUniqueUsers.append(uid)
                    validUser.append(uid)
                    validItem.append(items)
                    self.m_item = max(self.m_item, items)
                    self.n_user = max(self.n_user, uid)
                    self.validDataSize += 1
        self.validUniqueUsers = np.array(validUniqueUsers)
        self.validUser = np.array(validUser)
        self.validItem = np.array(validItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split('\t')
                    items = int(l[1])
                    uid = int(l[0])
                    if uid not in testUniqueUsers:
                        testUniqueUsers.append(uid)
                    testUser.append(uid)
                    testItem.append(items)
                    self.m_item = max(self.m_item, items)
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += 1
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        
        self.fairGraph = None
        self.semanticGraph = None
        self.userTail, self.user_tail_num = self.getTail(self.inter_info, inter_col='Inter_num', id_col='UserID', q=world.percentile)
        self.itemTail, self.item_tail_num = self.getTail(self.inter_info, inter_col='Inter_num', id_col='ItemID', q=world.percentile)


        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        # self.UserItemNet = csr_matrix((self.trainRating, (self.trainUser, self.trainItem)),
        #                               shape=(self.n_user, self.m_item))
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))


        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        self.build_group_info()
        self.Random_Walk()
        self.fair_reweight = FairGATReweighter(user_info=self.user_info, item_info=self.item_info, sen_attr=world.sen_attr,
                                               pi_u=self.attr_ratio, pi_i=self.genre_ratio, sen2idx=self.attr2idx, genre2idx=self.genre2idx,)



        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    @property
    def user_text_feature(self):
        return self._user_text_feature

    @property
    def item_text_feature(self):
        return self._item_text_feature

    def build_group_info(self):
        # self.u_cold = 5
        # self.i_cold = 'folk'
        self.u_cold = 0
        self.i_cold = 'Adventure'
        self.sen_attr = world.sen_attr
        # ========= 1) 用户敏感属性集合 & 映射（用 self.sen_attr） =========
        attr_col = self.sen_attr  # 例: 'Gender' / 'u_age_c' / 'AgeBucket'
        attrset = sorted(self.user_info[attr_col].dropna().unique().tolist())
        attr2idx = {a: i for i, a in enumerate(attrset)}

        # ========= 2) 物品类型集合 & 映射（全物品集） =========
        all_genres = self.item_info['Genres'].astype(str).str.split('|').explode()
        genreset = sorted(all_genres.dropna().unique().tolist())
        genre2idx = {g: i for i, g in enumerate(genreset)}

        # ========= 3) 用户敏感属性分布（全体用户集，不看训练是否有行为） =========
        attr_counts = (
            self.user_info[attr_col]
            .value_counts()
            .reindex(attrset, fill_value=0)
        )
        attr_ratio = (attr_counts / max(int(attr_counts.sum()), 1)).values  # 顺序对齐 attrset

        # ========= 4) 物品类型分布（全体物品集；多标签按出现次数计数） =========
        genre_counts = (
            self.item_info['Genres'].astype(str).str.split('|').explode()
            .value_counts()
            .reindex(genreset, fill_value=0)
        )
        genre_ratio = (genre_counts / max(int(genre_counts.sum()), 1)).values  # 顺序对齐 genreset

        # ========= 5) 保存到类属性 =========
        self.attr_col = attr_col
        self.attrset = attrset
        self.attr2idx = attr2idx
        self.attr_ratio = attr_ratio  # ← 全体用户分布（按敏感属性）

        self.genreset = genreset
        self.genre2idx = genre2idx
        self.genre_ratio = genre_ratio  # ← 全体物品分布（按类型）

        self.user_group = {}
        for row in self.user_info[['UserID', attr_col]].itertuples(index=False):
            uid, attr_val = row[0], row[1]
            idx = attr2idx.get(attr_val, -1)
            self.user_group[int(uid)] = int(idx)

        self.item_group = {}
        for row in self.item_info[['ItemID', 'Genres']].itertuples(index=False):
            iid, genres = int(row[0]), str(row[1]) if row[1] is not None else ''
            primary = genres.split('|')[0].strip() if genres else ''
            idx = genre2idx.get(primary, -1)
            self.item_group[iid] = int(idx)

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        if self.fairGraph is None:
            # self.Graph = self.fair_reweight()
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating fair adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                if world.RW_button == 0:
                    UserItemNet_Aug = self.graphEnhance()
                else:
                    print('without Random walk')
                    UserItemNet_Aug = self.UserItemNet

                if world.FR_button == 0:
                    R = self.fair_reweight.fair_reweight(UserItemNet=UserItemNet_Aug, theta=world.theta, phi=world.phi, gamma=world.gamma)
                else:
                    print('without fairReweight')
                    R = UserItemNet_Aug
                # UserItemNet_Aug = self.UserItemNet
                # R = self.fair_reweight(UserItemNet=UserItemNet_Aug).tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.tocsr()
                # adj_mat = adj_mat.todok()
                adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                norm_adj = adj_mat
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                # sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.fairGraph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.fairGraph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.fairGraph = self.fairGraph.coalesce().to(world.device)
                print("don't split the matrix")



        return self.fairGraph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getTail(self, df, inter_col='Inter_num', id_col='UserID', q=0.2):
        id_cnt = df[id_col].value_counts()  # Series: index 是 ID，值是出现次数

        # 2) 计算频次的分位数阈值
        thr = id_cnt.quantile(q)  # 底部 q 分位对应的频次

        # 3) 取出频次 <= 阈值的“长尾”ID
        tail_ids = id_cnt[id_cnt <= thr].index.tolist()

        return tail_ids, float(thr)

    # --- 工具：行归一 ---
    def _row_normalize(self, A):
        A = A.tocsr(copy=True)
        rs = np.asarray(A.sum(axis=1)).ravel() + EPS
        return sp.diags(1.0 / rs) @ A

    # --- k跳有效女性比例 rF^{(k)} ---
    def k_hop_female_ratio(self, A_ui, y_female01, k=0, norm='row'):
        """
        A_ui: csr [U,I]，用户->物品（0/1或权重）
        y_female01: ndarray [U]，女=1、男=0
        k: 传播层数；k=0 为“一跳比例”
        norm: 'row'（推荐）或 'none'
        return: rF^{(k)} ∈ [0,1]，shape=(I,)
        """
        assert sp.isspmatrix_csr(A_ui)
        U, I = A_ui.shape
        y = np.asarray(y_female01, dtype=float).reshape(-1, 1)
        ones = np.ones((U, 1), dtype=float)

        if norm == 'row':
            A_ui_hat = self._row_normalize(A_ui)
            A_iu_hat = self._row_normalize(A_ui_hat.T)
        elif norm == 'none':
            A_ui_hat = A_ui.tocsr()
            A_iu_hat = A_ui_hat.T.tocsr()
        else:
            raise ValueError("norm must be 'row' or 'none'")

        # S = (A_IU A_UI)^k A_IU ；逐次乘避免构大幂
        S = A_iu_hat.copy()
        B = A_iu_hat @ A_ui_hat  # |I| x |I|
        for _ in range(k):
            S = B @ S

        num = (S @ y).ravel()  # 到女性用户的到达强度
        den = (S @ ones).ravel() + EPS  # 到所有用户的到达强度
        return np.clip(num / den, 0.0, 1.0)

    # --- 物品相对基线女性倾向 ψ_i（方案A） ---
    def item_bias_logit_sigmoid(self, rF_item, pi_F, k_temp=1.0):
        r = np.clip(rF_item, EPS, 1 - EPS)
        pi = float(np.clip(pi_F, EPS, 1 - EPS))
        delta = np.log(r / (1 - r)) - np.log(pi / (1 - pi))
        return 1.0 / (1.0 + np.exp(-k_temp * delta))  # ψ_i ∈ [0,1]

    def Random_Walk(self):

        if os.path.exists(f'{self.path}/RW_adj_score_user.npy') and os.path.exists(f'{self.path}/RW_adj_score_item.npy'):
            print("Random Walk scores exist, skip computation.")
            return
        u_cold_ids = self.user_info['UserID'][self.user_info[self.sen_attr] == self.u_cold].tolist()
        i_cold_ids = self.item_info['ItemID'][self.item_info['Genres'] == self.i_cold].tolist()
        userid = self.inter_info['UserID'].values
        itemid = self.inter_info['ItemID'].values
        rating = self.inter_info['Rating'].values
        # inter_df = pd.read_csv(f'./{self.path}/inters.dat', header=None, names=["UserID", "ItemID", "Rating", "X","Y","Z"], sep='\t')
        # inter_df = pd.read_csv(f'./{self.path}/inters.dat', sep='\t')
        # userid1 = inter_df['u_id_c'].values
        # itemid1 = inter_df['i_id_c'].values
        # rating1 = inter_df['rating'].values
        inter_df = pd.read_csv(f'./{self.path}/inters.dat', sep='\t',header=None, names=["UserID", "ItemID", "Rating", "X","Y","Z"])
        userid1 = inter_df['UserID'].values
        itemid1 = inter_df['ItemID'].values
        rating1 = inter_df['Rating'].values
        rw = RW(userid, itemid, rating, userid1, itemid1, rating1,
                alpha=0.8, save_path=self.path, user_feat=self._user_text_feature, item_feat=self._item_text_feature,
                u_cold_ids=u_cold_ids, i_cold_ids=i_cold_ids, topk_u=10, topk_i=10)
        rw.get_adj_score(isuser=True)
        rw.get_adj_score(isuser=False)


    def graphEnhance(self):

        user_adj_score = np.load(f'{self.path}/wo_LLM_RW/RW_adj_score_user.npy',
                                 allow_pickle=True).item()
        item_adj_score = np.load(f'{self.path}/wo_LLM_RW/RW_adj_score_item.npy',
                                 allow_pickle=True).item()

        edge_set = set()
        edge = defaultdict(list)
        A = self.UserItemNet
        # score_dict = defaultdict(dict)
        for uid in self.userTail:
            u_neighbor_num = A.indptr[uid+1] - A.indptr[uid]
            us = user_adj_score['user_' + str(uid)]
            # f_num = int(self.user_tail_num - (self.m_item - len(us)))
            f_num = int(self.user_tail_num - u_neighbor_num)
            if f_num <= 0:
                continue
            dit = sorted(us.items(), key=lambda x: x[1], reverse=True)[:f_num]
            for it, sc in dit:
                iid = int(it[5:])
                if (uid, iid) in edge_set:
                    continue
                edge[uid].append(iid)
                edge_set.add((uid, iid))

        Ac = A.tocsc(copy=False)
        for iid in self.itemTail:

            iscor = item_adj_score['item_' + str(iid)]
            i_neighbor_num = Ac.indptr[iid+1] - Ac.indptr[iid]
            # f_num = int(self.item_tail_num - (self.n_user - len(iscor)))
            f_num = int(self.item_tail_num - i_neighbor_num)
            if f_num <= 0:
                continue
            dit = sorted(iscor.items(), key=lambda x: x[1], reverse=True)[:f_num]
            for it, sc in dit:
                uid = int(it[5:])
                if (uid, iid) in edge_set:
                    continue
                edge_set.add((uid, iid))
                edge[uid].append(iid)

        new_user = []
        new_item = []
        for uid, item_list in edge.items():
            for iid in item_list:
                new_user.append(uid)
                new_item.append(int(iid))  # 如果是字符串格式，需要转为 int

        new_data = np.ones(len(new_user), dtype=np.float32)

        # 2. 构造新的稀疏矩阵（同样是 [n_user, m_item]）
        new_edges = coo_matrix((new_data, (new_user, new_item)), shape=(self.n_user, self.m_item)).tocsr()

        # 3. 合并原图与新图（注意防止边权相加超过1）
        UserItemNet_aug = self.UserItemNet + new_edges
        UserItemNet_aug.data = np.clip(UserItemNet_aug.data, 0, 1)

        return UserItemNet_aug

class RW():
    def __init__(self, X, Y, R, X1, Y1, R1,
                 alpha, save_path, user_feat, item_feat,
                 u_cold_ids, i_cold_ids, topk_u, topk_i):
        XX, YY = ['user_' + str(x) for x in X], ['item_' + str(y) for y in Y]

        # print(X)
        # print(Y)
        # C, D = [], []
        CC, DD, RR = [], [], []
        for x, y, r in zip(X1, Y1, R1):
            if x in u_cold_ids or y in i_cold_ids:
                if random.random() < 0.6:
                    continue
            CC.append(x);DD.append(y);RR.append(r)

        XX1, YY1 = ['user_' + str(x) for x in CC], ['item_' + str(y) for y in DD]

        # C, D, RR = np.array(C), np.array(D), np.array(RR)

        self.save_path = save_path
        self.user_num = max(X.max(), max(u_cold_ids)) + 1
        self.item_num = max(Y.max(), max(i_cold_ids)) + 1
        self.uimatrix = csr_matrix((RR, (CC, DD)), shape=(self.user_num, self.item_num))
        self.uimatrix1 = csr_matrix((R, (X, Y)), shape=(self.user_num, self.item_num))
        self.usersum = np.array(self.uimatrix.sum(axis=1), dtype=float).squeeze()
        self.itemsum = np.array(self.uimatrix.sum(axis=0), dtype=float).squeeze()
        self.alpha = alpha
        # print(self.uimatrix[1,661])3
        self.QG = self.get_graph(XX1, YY1, 0)
        self.QG1 = self.get_graph(XX, YY, 1)

        # self.QG = self.get_graph(XX, YY, 0)

        uu_edges, ii_edges = self._build_uu_ii_edges(user_feat, item_feat, topk_u, topk_i, cold_item_ids=i_cold_ids, cold_user_ids=u_cold_ids)
        self._merge_aux_edges(uu_edges, ii_edges)
        # self.QG1 = self.get_graph(XX1, YY1, 1)
        # item_51，item_91，item_109,item_115，142，143没有邻居节点0
        m, vertex, address_dict = self.graph_to_m(self.QG)
        self.m = m
        self.vertex = vertex
        self.address_dict = address_dict
        mat_all = self.mat_all_point(m, vertex, self.alpha)
        self.mat_all = mat_all

    # --------- 只对指定“源行”做 top-k 的余弦相似 ---------
    def _cos_topk_dense_rows(self, M, src_rows, topk, lam=1.0, label='user'):
        """
        只从 src_rows 这些行发出单向 top-k 相似边，指向全集 (包含热/冷)。
        返回: {src_idx: {dst_idx: lam*cos(src,dst)}}
        """
        N = M.shape[0]
        M = M.detach().cpu().numpy()
        if src_rows is None:
            src_rows = np.arange(N, dtype=int)
        else:
            src_rows = np.asarray(src_rows, dtype=int)
            if src_rows.size == 0:
                return {}

        X = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)  # [N,d]
        out = {}
        for i in src_rows:
            s = X[i] @ X.T  # [N,]
            s[i] = -np.inf  # 去自环
            if topk < N:
                idx = np.argpartition(-s, kth=min(topk - 1, N - 1))[:topk]
            else:
                idx = np.arange(N, dtype=int)
            vals = s[idx] * 5.0 * 2.5
            # 清理 -inf / nan
            mask = np.isfinite(vals)
            idx, vals = idx[mask], vals[mask]
            if idx.size == 0:
                continue
            order = np.argsort(-vals)
            idx, vals = idx[order], vals[order]
            out[int(i)] = {int(j): float(lam * v) for j, v in zip(idx, vals)}
            if label == 'user':
                self.usersum[i] += sum(vals)
            elif label == 'item':
                self.itemsum[i] += sum(vals)
        return out

    def _build_uu_ii_edges(self, user_feat, item_feat, topk_u, topk_i,
                           cold_user_ids=None, cold_item_ids=None):
        """
        只对冷用户(cold_user_ids)与冷物品(cold_item_ids)发出相似边。
        若特征为空，则退化为共现余弦(仅 warm 能算到)；冷且无交互时可能采不到邻居 → 建议提供内容向量。
        """
        # ---- 用户侧 ----
        if user_feat is not None:
            uu_edges = self._cos_topk_dense_rows(user_feat, cold_user_ids, topk=topk_u, label='user')
        else:
            # 共现余弦：U = A A^T / (||u||·||v||)
            A = self.uimatrix.toarray()
            U = A @ A.T
            du = np.sqrt(self.usersum + 1e-8)
            S = U / (du[:, None] * du[None, :] + 1e-8)
            uu_edges = self._cos_topk_dense_rows(S, cold_user_ids, topk=topk_u)

        # ---- 物品侧 ----
        if item_feat is not None:
            ii_edges = self._cos_topk_dense_rows(item_feat, cold_item_ids, topk=topk_i, label='item')
        else:
            A = self.uimatrix.toarray()
            I = A.T @ A
            di = np.sqrt(self.itemsum + 1e-8)
            S = I / (di[:, None] * di[None, :] + 1e-8)
            ii_edges = self._cos_topk_dense_rows(S, cold_item_ids, topk=topk_i)

        return uu_edges, ii_edges

    def _merge_aux_edges(self, uu_edges, ii_edges):
        """把 U–U / I–I 边并入 QG（单向，不强制对称），权重已含系数 lam_u/lam_i。"""
        # U–U
        for u, nbrs in uu_edges.items():
            uk = f'user_{u}'
            self.QG.setdefault(uk, {})
            for v, w in nbrs.items():
                vk = f'user_{v}'
                if u == v:
                    continue
                # self.QG.setdefault(vk, {})
                self.QG[uk][vk] = self.QG[uk].get(vk, 0.0) + float(w)
        # I–I
        for i, nbrs in ii_edges.items():
            ik = f'item_{i}'
            self.QG.setdefault(ik, {})
            for j, w in nbrs.items():
                jk = f'item_{j}'
                if i == j:
                    continue
                # self.QG.setdefault(jk, {})
                self.QG[ik][jk] = self.QG[ik].get(jk, 0.0) + float(w)

    def get_graph(self, X, Y, l):
        """
        Args:
            X: user id
            Y: item id
        Returns:
            graph:dic['user_id1':{'item_id1':1},  ... ]
        """
        if l == 0:
            uimatrix = self.uimatrix

        elif l == 1:
            uimatrix = self.uimatrix1


        item_user = dict()
        for i in range(len(X)):
            user = X[i]
            item = Y[i]
            if item not in item_user:
                item_user[item] = {}
            # item_user[item][user]=1
            item_user[item][user] = uimatrix[int(user[5:]), int(item[5:])]
        # print(item_user)

        user_item = dict()
        for i in range(len(Y)):
            user = X[i]
            item = Y[i]
            if user not in user_item:
                user_item[user] = {}
            # user_item[user][item]=1
            user_item[user][item] = uimatrix[int(user[5:]), int(item[5:])]
        # print(user_item)
        G = dict(item_user, **user_item)
        # np.save('../data/lastfm/RW_adj_score/u_i_adj.npy',user_item)
        # np.save('../data/lastfm/RW_adj_score/i_u_adj.npy', item_user)
        # print('Gwancheng')
        # print(G)
        return G

    def get_ego_graph(self, uid):

        G = copy.deepcopy(self.QG)
        user = 'user_' + str(uid)
        user_item = dict()
        user_item[user] = copy.deepcopy(G[user])
        item_user = dict()
        items = copy.deepcopy(G[user])
        begin = time.time()
        for it in items.keys():  # 遍历一阶邻居节点item_id
            if it not in item_user:
                item_user[it] = copy.deepcopy(G[it])
        for it in item_user.keys():
            for us in item_user[it].keys():
                if us not in user_item:
                    user_item[us] = copy.deepcopy(G[us])  # 一阶邻居周围的用户加入自中心图
                for i in list(user_item[us].keys()):
                    if i not in item_user:
                        del user_item[us][i]
        # for us in user_item.keys():
        #     for it in user_item[us].keys():
        #         if it not in item_user:
        #             item_user[it]=G[it]#一阶邻居周围用户的周围物品加入自中心图
        EG = dict(item_user, **user_item)
        end = time.time()
        print('egotime', end - begin)
        # print(G['user_1'])
        return EG
    # def generate_ego_graph(self):
    #     for userID in range(user_num):
    #         print(userID)
    #         id=userID+1
    #         eg=self.get_ego_graph(id)
    #         self.e_graph.append(eg)
    #         gc.collect()
    #     np.save('ego_graph/user_ego_graph.npy',self.e_graph)
    def RW_in_ego_graph(self, alpha, userID, max_depth):
        # rank = dict()
        G = self.get_ego_graph(userID)
        userID = 'user_' + str(userID)
        rank = {x: 0 for x in G.keys()}
        rank[userID] = 1
        # 开始迭代
        # begin = time.time()
        for k in range(max_depth):
            tmp = {x: 0 for x in G.keys()}
            # 取出节点i和他的出边尾节点集合ri
            for i, ri in G.items():
                # 取节点i的出边的尾节点j以及边E(i,j)的权重wij,边的权重都为1，归一化后就是1/len(ri)
                for j, wij in ri.items():
                    tmp[j] += alpha * rank[i] / (1.0 * len(ri))
            tmp[userID] += (1 - alpha)
            rank = tmp
        print(rank)
        # end = time.time()
        # print('use_time', end - begin)
        lst = sorted(rank.items(), key=lambda x: x[1], reverse=True)[:10]  # 排序
        for ele in lst:
            print("%s:%.3f, \t" % (ele[0], ele[1]))

    def graph_to_m(self, G):
        """
        Returns:
            a coo_matrix sparse mat M
            a list,total user item points
            a dict,map all the point to row index
        """
        graph = G
        vertex = list(graph.keys())
        address_dict = {}
        total_len = len(vertex)
        for index in range(len(vertex)):
            address_dict[vertex[index]] = index
        # np.save('../data/lastfm/RW_adj_score/address_dict.npy',address_dict)
        # print('wancehng')
        row = []
        col = []
        data = []
        for element_i in graph:
            # weight = round(1/len(graph[element_i]),3)####取下一个节点的概率
            if element_i[0] == 'u':
                sum = self.usersum[int(element_i[5:])]
            else:
                sum = self.itemsum[int(element_i[5:])]
            row_index = address_dict[element_i]
            for element_j in graph[element_i]:
                col_index = address_dict[element_j]
                row.append(row_index)
                col.append(col_index)
                # data.append(weight)
                x = graph[element_i][element_j]
                # x=float(x)
                x = x / sum
                data.append(x)  # 按评分的概率分布
        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        m = coo_matrix((data, (row, col)), shape=(total_len, total_len))
        return m, vertex, address_dict

    def mat_all_point(self, m_mat, vertex, alpha):
        """
        get E-alpha*m_mat.T
        Args:
            m_mat
            vertex:total item and user points
            alpha:the prob for random walking
        Returns:
            a sparse
        """

        total_len = len(vertex)
        row = []
        col = []
        data = []
        for index in range(total_len):
            row.append(index)
            col.append(index)
            data.append(1)
        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        eye_t = coo_matrix((data, (row, col)), shape=(total_len, total_len))

        return eye_t.tocsr() - alpha * m_mat.tocsr().transpose()

    def RW_use_matrix(self, ID, isuser, K=10, use_matrix=True):
        """
        Args:
            alpha:the prob for random walking
            userID:the user to recom
            K:recom item num
        Returns:
            a dic,key:itemid ,value:pr score
        """
        G = self.QG

        # m, vertex, address_dict = self.graph_to_m(G)
        m = self.m
        vertex = self.vertex
        address_dict = self.address_dict
        # userID = 'item_' + str(ID)
        if isuser:
            stance = 'user_' + str(ID)
        else:
            stance = 'item_' + str(ID)

        # print('add',address_dict)
        if stance not in address_dict:
            return []
        score_dict = {}
        recom_dict = {}
        mat_all = self.mat_all
        index = address_dict[stance]
        initial_list = [[0] for row in range(len(vertex))]
        initial_list[index] = [1]
        r_zero = np.array(initial_list)
        # r_zero=np.zeros(shape=(5100,1),dtype=int)
        # r_zero[0,0]=1
        # r_zero[1,1]=1
        # res = gmres(mat_all,r_zero,tol=1e-8)[0]
        # print(mat_all.shape)
        # np.save('../data/lastfm/RW_adj_score/mat_all.npy', mat_all)
        t0 = time()
        # res=cgs(mat_all,r_zero)

        res = gmres(mat_all, r_zero, tol=1e-8)[0]
        t1 = time()
        print(t1 - t0)
        print(res.shape)
        # mat_all=mat_all.tocsc()
        # A=inv(mat_all)
        # res=A[:,index]
        for index in range(len(res)):
            point = vertex[index]
            if len(point.strip().split('_')) < 2:
                continue
            # if point in G[userID]:
            #     continue
            score_dict[point] = res[index]
        # print(score_dict)
        # for it in list(score_dict.keys()):
        #     if score_dict[it] <va :#去掉噪声节点
        #         del score_dict[it]
        # for zuhe in sorted(score_dict.items(),key=operator.itemgetter(1),reverse=True)[:K]:
        #     point,score = zuhe[0],zuhe[1]
        #     recom_dict[point] = score
        # print(recom_dict)
        return score_dict

    def get_adj_score(self, isuser):
        if isuser:
            user_adj_dict = {}
            for ID in range(self.user_num):
                print(ID)
                user = 'user_' + str(ID)
                user_adj_dict[user] = {}

                score_dict = self.RW_use_matrix(ID, isuser)
                # print(f'保存成功，路径：{save_path}')
                if not score_dict:
                    continue
                # stance = 'user_' + str(ID)
                one_order = self.QG[user].keys()
                # one_order = self.QG1[user].keys()
                filtered_iter = (
                    (it, score) for it, score in score_dict.items()
                    if (it not in one_order) and it.startswith('item_')
                )
                topk_pairs = heapq.nlargest(100, filtered_iter, key=lambda kv: kv[1])
                for it, score in topk_pairs:
                    item_id = int(it[5:])  # 'item_' 之后
                    user_adj_dict[user][f'item_{item_id}'] = score
                # for it, score in score_dict.items():
                #     if it in one_order:
                #         continue
                #     if not it.startswith('item_'):
                #         continue  # 只保留 item 节点
                #     item_id = int(it[5:])
                #     user_adj_dict[user][f'item_{item_id}'] = score
            np.save(f'{self.save_path}/RW_adj_score_user.npy', user_adj_dict)
        else:
            item_adj_dict = {}
            for ID in range(self.item_num):
                print(ID)
                item = 'item_' + str(ID)
                item_adj_dict[item] = {}
                score_dict = self.RW_use_matrix(ID, isuser)
                if not score_dict:
                    continue
                # stance = 'item_' + str(ID)
                try:
                    one_order = self.QG[item].keys()
                    # one_order = self.QG1[item].keys()
                except KeyError:
                    print(item)
                    continue
                filtered_iter = (
                    (it, score) for it, score in score_dict.items()
                    if (it not in one_order) and it.startswith('user_')
                )
                topk_pairs = heapq.nlargest(100, filtered_iter, key=lambda kv: kv[1])

                for it, score in topk_pairs:
                    user_id = int(it[5:])  # 'user_' 之后
                    item_adj_dict[item][f'user_{user_id}'] = score
                # for it, score in score_dict.items():
                #     if it in one_order:
                #         continue
                #     if not it.startswith('user_'):
                #         continue
                #     user_id = int(it[5:])
                #     item_adj_dict[item][f'user_{user_id}'] = score
            np.save(f'{self.save_path}/RW_adj_score_item.npy', item_adj_dict)

import numpy as np
import pandas as pd
import scipy.sparse as sp

EPS = 1e-12


class FairGATReweighter:
    """
    k-hop 环境统计 + Dirichlet 平滑/基线收缩 + 乘法门控（对称、可托管到 GAT）
    这里只返回 P（行随机转移），若后续要给 GAT 加 edge-bias，可在构建 a_data 后取 log(a_data) 使用。
    """

    def __init__(self, user_info: pd.DataFrame, item_info: pd.DataFrame, sen_attr: str, pi_u=None, pi_i=None, sen2idx=None, genre2idx=None):
        self.user_info = user_info.reset_index(drop=True)
        self.item_info = item_info.reset_index(drop=True)
        self.sen2idx = sen2idx
        self.genre2idx = genre2idx
        self.pi_u = pi_u
        self.pi_i = pi_i
        self.sen_attr = sen_attr

    # ---------- 基础工具 ----------
    @staticmethod
    def row_normalize(A: sp.csr_matrix) -> sp.csr_matrix:
        A = A.tocsr(copy=True)
        rs = np.asarray(A.sum(axis=1)).ravel()
        nz = rs > 0
        rs[nz] = 1.0 / rs[nz]
        return sp.diags(rs) @ A

    @staticmethod
    def jsd01(p: np.ndarray, q: np.ndarray) -> float:
        """JSD(p||q)/log2 ∈ [0,1]；p,q 已是概率向量"""
        p = np.clip(p.astype(float), EPS, 1.0); p /= p.sum()
        q = np.clip(q.astype(float), EPS, 1.0); q /= q.sum()
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * (np.log(p) - np.log(m)))
        kl_qm = np.sum(q * (np.log(q) - np.log(m)))
        return 0.5 * (kl_pm + kl_qm) / np.log(2.0)

    @staticmethod
    def kl_div(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> float:
        """
        KL(p||q)；p,q 已为非负向量（会在函数里归一化）
        """
        p = np.clip(p.astype(float), eps, None)
        p /= p.sum()
        q = np.clip(q.astype(float), eps, None)
        q /= q.sum()
        return float(np.sum(p * (np.log(p) - np.log(q))))

    # ---------- 取标签并编码 ----------
    def _extract_labels(self):
        """
        从 DataFrame 提取并编码：
          user_group: [U] 性别 M→0, F→1
          item_type : [I] Genres 映射到 0..C-1（多标签取第1个为主标签）
          group_names, type_names: 便于日志
        """
        # 用户性别
        user_group = np.asarray([self.sen2idx[g] for g in self.user_info[self.sen_attr]])
        # 物品类型（若是多标签 "Action|Thriller"，默认取第一个）
        item_type = np.asarray([self.genre2idx[g] for g in self.item_info['Genres']])

        return user_group, item_type

    # ---------- k-hop 计数：只用一维标签 ----------
    def k_hop_counts_from_labels(
        self,
        A_ui: sp.csr_matrix,
        labels: np.ndarray,         # side='user' 传 item_type[I]；side='item' 传 user_group[U]
        side: str,                  # 'user' -> 返回 U×C；'item' -> 返回 I×G
        K: int = 1,
        hop_weights=None,
        norm: str = 'row',
        n_classes: int = None,
    ) -> np.ndarray:
        """
        叠加 1/3/5... hop 的“邻居平均”计数（非归一），后续用于平滑。
        K=1 -> 叠加 1/3 hop；K=2 -> 叠加 1/3/5 hop。
        """
        assert sp.isspmatrix_csr(A_ui)
        U, I = A_ui.shape
        labels = np.asarray(labels, dtype=int)
        if n_classes is None:
            n_classes = int(labels.max()) + 1

        # 归一化选择
        if norm == 'row':
            A_ui_hat = self.row_normalize(A_ui)
            A_iu_hat = self.row_normalize(A_ui_hat.T)
        elif norm == 'none':
            A_ui_hat = A_ui.tocsr(); A_iu_hat = A_ui_hat.T.tocsr()
        else:
            raise ValueError("norm must be 'row' or 'none'")

        # hop 权重（近邻更大）
        if hop_weights is None:
            base = np.array([0.7, 0.2, 0.1], dtype=float)
            hop_weights = base[:K+1]
        hop_weights = hop_weights / (hop_weights.sum() + EPS)

        if side == 'user':
            # I×C 指示稀疏矩阵
            Y = sp.csr_matrix((np.ones(I), (np.arange(I), labels)), shape=(I, n_classes))
            S = A_ui_hat.copy()             # U×I
            B = A_ui_hat @ A_iu_hat         # U×U
            acc = (S @ Y) * hop_weights[0]  # U×C
            for t in range(1, K + 1):
                S = B @ S
                acc = acc + (S @ Y) * hop_weights[t]
            return acc.toarray()

        elif side == 'item':
            # U×G 指示稀疏矩阵
            X = sp.csr_matrix((np.ones(U), (np.arange(U), labels)), shape=(U, n_classes))
            S = A_iu_hat.copy()             # I×U
            B = A_iu_hat @ A_ui_hat         # I×I
            acc = (S @ X) * hop_weights[0]  # I×G
            for t in range(1, K + 1):
                S = B @ S
                acc = acc + (S @ X) * hop_weights[t]
            return acc.toarray()

        else:
            raise ValueError("side must be 'user' or 'item'")

    # ---------- Dirichlet 平滑 + 基线收缩 ----------
    @staticmethod
    def smooth_and_shrink(counts: np.ndarray, pi: np.ndarray, tau: float = 10.0, rho: float = 0.7) -> np.ndarray:
        pi = np.asarray(pi, dtype=float); pi /= (pi.sum() + EPS)
        sums = counts.sum(axis=1, keepdims=True)
        post = (counts + tau * pi[None, :]) / (sums + tau + EPS)   # Dirichlet
        out  = (1 - rho) * pi[None, :] + rho * post                # 收缩
        return out

    # ---------- 边级门控：S_t·C_t·S_g·C_g ----------
    def build_edge_gates_singleknob(self,
                                    UI: sp.csr_matrix,
                                    user_group: np.ndarray,  # [U]
                                    item_type: np.ndarray,  # [I]
                                    q_u: np.ndarray,  # U×C（已是分布）
                                    p_i: np.ndarray,  # I×G（已是分布）
                                    pi_c: np.ndarray,
                                    pi_g: np.ndarray,
                                    theta: float = 1.0,
                                    phi: float = 1.0,
                                    gamma: float = 0.8,
                                    divergence: str = "jsd"  # 'jsd' | 'tv'
                                    ) -> np.ndarray:
        """
        方案1·一把闸：
          s_{ui} = -theta * ( ln A_g + ln A_c ) - phi * ( d_g + d_t )
          a_{ui} = exp( gamma * tanh( s_{ui} ) ) ∈ [e^{-gamma}, e^{gamma}]
        其中：
          A_g = p_i[i, g(u)] / pi_g[g(u)]
          A_c = q_u[u, c(i)] / pi_c[c(i)]
          d_g, d_t 分别为“去敏感维”后的形状差异（JSD 或 TV）
        """
        U, I = UI.shape
        indptr, indices = UI.indptr, UI.indices
        a_out = np.empty_like(UI.data, dtype=float)

        G = int(user_group.max()) + 1
        C = None if item_type is None else int(item_type.max()) + 1

        for u in range(U):
            s, e = indptr[u], indptr[u + 1]
            if s == e:
                continue
            nbr = indices[s:e]
            gu = int(user_group[u])

            # ---- A_g(u,i) 与 ln A_g
            Ag = p_i[nbr, gu] / (pi_g[gu] + EPS)
            lnAg = np.log(np.clip(Ag, EPS, None))

            # ---- A_c(u,i) 与 ln A_c
            if item_type is not None and q_u is not None:
                ci = item_type[nbr]  # 每条边对应的类型
                Ac = q_u[u, ci] / (pi_c[ci] + EPS)
                lnAc = np.log(np.clip(Ac, EPS, None))
            else:
                lnAc = 0.0
                ci = None

            # ---- d_g（去掉 gu）
            if G is not None and G >= 3:
                pi_g_rest = pi_g.copy()
                pi_g_rest[gu] = 0.0
                pi_g_rest = pi_g_rest / (pi_g_rest.sum() + EPS)
                # 邻居逐个计算
                dg = np.empty(e - s, dtype=float)
                for k, i in enumerate(nbr):
                    vec = p_i[i, :].copy()
                    vec[gu] = 0.0
                    vec = vec / (vec.sum() + EPS)
                    dg[k] = self.kl_div(vec, pi_g_rest)
            else:
                dg = np.zeros(e - s, dtype=float)

            # ---- d_t（去掉 ci）
            if ci is not None and C >= 3:
                dt = np.empty(e - s, dtype=float)
                for k in range(e - s):
                    t = int(ci[k])
                    qrest = q_u[u, :].copy()
                    qrest[t] = 0.0
                    qrest = qrest / (qrest.sum() + EPS)
                    pirest = pi_c.copy()
                    pirest[t] = 0.0
                    pirest = pirest / (pirest.sum() + EPS)
                    dt[k] = self.kl_div(qrest, pirest)
            else:
                dt = np.zeros(e - s, dtype=float)

            # ---- s 与 a
            s_ui = -theta * (lnAg + lnAc) - phi * (dg + dt)
            a_out[s:e] = np.exp(gamma * np.tanh(s_ui))  # ∈ [e^{-γ}, e^{γ}]

        return a_out

    # ---------- 主入口：只返回 P ----------
    def fair_reweight(self,
                      UserItemNet: sp.csr_matrix,
                      K: int = 1, hop_weights=None, norm='row',
                      # —— 三个强度（一把闸） —— #
                      theta: float = 1.0,  # 过曝降/欠曝升力度
                      phi: float = 1.0,  # 其余维“偏形”惩罚力度
                      gamma: float = 0.8,  # 最大放大/缩小幅度：[e^{-γ}, e^{γ}]
                      # 差异度量
                      divergence: str = "jsd"  # 'jsd' | 'tv'
                      ) -> sp.csr_matrix:
        """
        返回 P: csr [U,I] —— 重加权并行归一后的用户->物品转移矩阵（每行和=1）
        K=1：叠加 1/3-hop；K=2：叠加 1/3/5-hop（一般 K=1 足够）
        """
        UI = UserItemNet.tocsr(copy=True)
        if UI.data is None or UI.data.size == 0:
            UI.data = np.ones_like(UI.indices, dtype=float)

        # 1) 标签
        user_group, item_type = self._extract_labels()
        U, I = UI.shape
        G = int(user_group.max()) + 1
        C = int(item_type.max()) + 1
        # 2) 基线（度加权）
        pi_g = self.pi_u
        pi_c = self.pi_i

        # 3) k-hop 非归一化计数（只做邻居平均，不做平滑/收缩）
        P_counts = self.k_hop_counts_from_labels(UI, user_group, side='item',
                                                 K=K, hop_weights=hop_weights, norm=norm)  # I×G
        Q_counts = self.k_hop_counts_from_labels(UI, item_type, side='user',
                                                 K=K, hop_weights=hop_weights, norm=norm)  # U×C
        # 归一化为分布
        p_i = P_counts / (P_counts.sum(axis=1, keepdims=True) + EPS)  # I×G
        q_u = Q_counts / (Q_counts.sum(axis=1, keepdims=True) + EPS)  # U×C

        # 4) 逐边门控（新：对数线性→tanh→指数）
        a_data = self.build_edge_gates_singleknob(UI, user_group, item_type,
                                                  q_u, p_i, pi_c, pi_g,
                                                  theta=theta, phi=phi, gamma=gamma,
                                                  divergence=divergence)

        # 5) 乘回原权重并行归一
        new_data = UI.data * a_data
        P = sp.csr_matrix((new_data, UI.indices, UI.indptr), shape=UI.shape)
        P = self.row_normalize(P)
        return P

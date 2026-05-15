"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch

from dataloader import BasicDataset
from torch import nn

from torch_geometric.nn import GATv2Conv, GATConv


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    def fair_loss(self, users):
        """
        Parameters:
            users: users list
        Return:
            fair loss
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : BasicDataset = dataset
        self.__init_weight()
        self.__init_component()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        # self.userTail = self.dataset.userTail
        # self.itemTail = self.dataset.itemTail
        self.text_dim = 384
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']

        self.A_split = self.config['A_split']
        self.gat_heads = self.config['gat_heads']
        self.fair_k = self.config['fair_k']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.user_preference_text = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.item_preference_text = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim
        )

        self.item_text_embedding = self.dataset.item_text_feature
        self.user_text_embedding = self.dataset.user_text_feature

        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
#             nn.init.normal_(self.embedding_user.weight, std=0.1)
#             nn.init.normal_(self.embedding_item.weight, std=0.1)
            nn.init.xavier_uniform_(self.embedding_user.weight)
            nn.init.xavier_uniform_(self.embedding_item.weight)
            nn.init.xavier_uniform_(self.user_preference_text.weight)
            nn.init.xavier_uniform_(self.item_preference_text.weight)
            world.cprint('use UNIFORM distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        # self.Graph = self.dataset.getSparseGraph()
        self.fairGraph = self.dataset.getSparseGraph()
        self.edge_index, self.edge_weight = self._graph_to_edge()
        # self.semanticGraph = self.dataset.getSemanticGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")

    def __init_component(self):

        self.id_proj_user = nn.Identity()
        self.id_proj_item = nn.Identity()

        self.Fusion1 = EmbeddingTwoSemantic(self.latent_dim)
        self.Fusion2 = EmbeddingTwoSemantic(self.latent_dim)

        self.text_proj_user = nn.Sequential(
            nn.Linear(self.text_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim)
        )  # 降维部分
        self.text_proj_item = nn.Sequential(
            nn.Linear(self.text_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim)
        )

        self.gat_layers_v2 = torch.nn.ModuleList([
            GATv2Conv(
                in_channels=self.latent_dim,
                out_channels=self.latent_dim,
                heads=self.gat_heads,
                concat=False,  # 多头平均，维度不变
                dropout=0.1,
                edge_dim=1,  # ★ 关键：启用边特征
                add_self_loops=False  # 你的图已有权重时，通常不要再自动加自环
            )
            for _ in range(self.n_layers)
        ])

        self.gat_layers = torch.nn.ModuleList([
            GATConv(
                in_channels=self.latent_dim,
                out_channels=self.latent_dim,
                heads=self.gat_heads,
                concat=False,  # 多头平均，维度不变
                dropout=0.1,
                edge_dim=1,  # ★ 关键：启用边特征
                add_self_loops=False  # 你的图已有权重时，通常不要再自动加自环
            )
            for _ in range(self.n_layers)
        ])

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """

        id_emb_users = self.embedding_user.weight
        id_emb_items = self.embedding_item.weight
        id_emb_users = self.id_proj_user(id_emb_users)
        id_emb_items = self.id_proj_item(id_emb_items)

        users = id_emb_users
        items = id_emb_items


        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.fairGraph
        else:
            g_droped = self.fairGraph

        # users, items = self.Agg(users, items, g_droped)
        users, items = self.Agg_gatv2(users, items)

        return users, items

    def _graph_to_edge(self):
        g = self.fairGraph.coalesce()
        edge_index = g.indices().long()
        edge_weight = g.values().float()
        clamp = 1e-12
        kappa = 1.0
        # edge_attr = torch.log(edge_weight.clamp_min(clamp)).mul_(kappa)  # [E,1]

        return edge_index, edge_weight

    def Agg(self, user_emb, item_emb, graph, agg_fn=None):
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        # all_id_emb = torch.cat([id_emb_users, id_emb_items], dim=0)
        embs = [all_emb]
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(graph)):
                    temp_emb.append(torch.sparse.mm(graph[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(graph, all_emb)
            embs.append(all_emb)
        if agg_fn is not None:
            light_out = agg_fn(embs)
        else:
            embs = torch.stack(embs, dim=1)
            light_out = torch.mean(embs, dim=1)

        users, items = torch.split(light_out, [self.num_users, self.num_items])

        return users, items

    def Agg_gatv2(self, user_emb, item_emb, agg_fn=None):
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        embs = [all_emb]
        for gat in self.gat_layers_v2:
            all_emb = gat(all_emb, self.edge_index, edge_attr=self.edge_weight)
            embs.append(all_emb)

        if agg_fn is not None:
            light_out = agg_fn(embs)
        else:
            embs = torch.stack(embs, dim=1)
            light_out = torch.mean(embs, dim=1)

        users, items = torch.split(light_out, [self.num_users, self.num_items])

        return users, items

    def Agg_gat(self, user_emb, item_emb, agg_fn=None):
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        embs = [all_emb]
        for gat in self.gat_layers:
            all_emb = gat(all_emb, self.edge_index, edge_attr=self.edge_weight)
            embs.append(all_emb)

        if agg_fn is not None:
            light_out = agg_fn(embs)
        else:
            embs = torch.stack(embs, dim=1)
            light_out = torch.mean(embs, dim=1)

        users, items = torch.split(light_out, [self.num_users, self.num_items])

        return users, items


    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        # (users_emb, pos_emb, neg_emb,
        # userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos]
        neg_emb = all_items[neg]
        userEmb0 = self.embedding_user(users)
        posEmb0 = self.embedding_item(pos)
        negEmb0 = self.embedding_item(neg)

        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss

    def forward(self, users, items, graph=None):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma

class IdFusionModel(nn.Module):
    def __init__(self, in_dim, embed_dim): # tid, iid, text, image
        super(IdFusionModel, self).__init__()
        self.text2item = nn.Sequential(
            nn.Linear(in_dim, embed_dim * 3),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(embed_dim * 3, embed_dim)
        )  # 降维部分
        self.text2user = nn.Sequential(
            nn.Linear(in_dim, embed_dim * 3),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(embed_dim * 3, embed_dim)
        )

        self.id2item = nn.Linear(embed_dim, embed_dim)
        self.id2user = nn.Linear(embed_dim, embed_dim)

        self.text_item = EmbeddingTwoSemantic(embed_dim)
        self.text_user = EmbeddingTwoSemantic(embed_dim)

        self.item_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.user_norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, user_embedding, item_embedding, user_text_embedding, item_text_embedding):
        user_text = self.text2user(user_text_embedding)  # 对齐user_text和item_id
        item_text = self.text2item(item_text_embedding)  #对齐 item_text和item_id

        item = self.id2item(item_embedding)
        user = self.id2user(user_embedding)

        user_text = self.user_norm(user_text)
        user = self.user_norm(user)

        item_text = self.item_norm(item_text)
        item = self.item_norm(item)

        user_specific_text = self.text_user(user_text, user)
        item_specific_text = self.text_item(item_text, item)
        # item_fusion_embedding = self.fusion_item(item_specific_image, item_specific_text)

        return user_specific_text, item_specific_text

    def get_text(self):
        with torch.no_grad():
            text = self.text2item(self.text_embedding)
        return text

    def get_image(self):
        with torch.no_grad():
            image = self.image2item(self.image_embedding)
        return image

class EmbeddingTwoSemantic(nn.Module):
    def __init__(self, embedding_dim):
        super(EmbeddingTwoSemantic, self).__init__()
        self.query = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_feature_list):
        # embedding1 = input_feature1
        # embedding2 = input_feature2
        # embedding3 = input_feature3
        att = torch.cat([self.query(embedding) for embedding in input_feature_list], dim=-1)
        weight = self.softmax(att)
        h = sum(weight[:, i].unsqueeze(dim=1) * embedding for i, embedding in enumerate(input_feature_list))

        return h

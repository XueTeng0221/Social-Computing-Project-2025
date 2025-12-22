# models.py

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATConv, Linear

class FraudDetector(nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata, text_embed_dim=768):
        """
        参数:
        - text_embed_dim: 输入的文本向量维度 (RoBERTa 默认为 768)
        """
        super(FraudDetector, self).__init__()
        # 将 BERT 的 768 维向量 映射到 GNN 的 hidden_channels
        self.post_proj = Linear(text_embed_dim, hidden_channels)
        self.user_proj = None
        self.entity_proj = None
        self.convs = nn.ModuleList()
        for _ in range(2):
            conv_dict = {
                ('user', 'publish', 'post'): GATConv((-1, -1), hidden_channels, heads=2, concat=False, add_self_loops=False),
                ('user', 'repost', 'post'): GATConv((-1, -1), hidden_channels, heads=2, concat=False, add_self_loops=False),
                ('post', 'contain', 'entity'): GATConv((-1, -1), hidden_channels, heads=2, concat=False, add_self_loops=False),
                ('user', 'interact', 'user'): GATConv((-1, -1), hidden_channels, heads=2, concat=False, add_self_loops=False),
                ('user', 'follow', 'user'): GATConv((-1, -1), hidden_channels, heads=2, concat=False, add_self_loops=False),
                ('post', 'similar', 'post'): GATConv((-1, -1), hidden_channels, heads=2, concat=False, add_self_loops=False),
            }
            conv = HeteroConv(conv_dict, aggr='sum')
            self.convs.append(conv)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, out_channels)
        )
    
    def forward(self, x_dict, edge_index_dict):
        # ========== 1. 节点特征提取 ==========
        
        # 修改点：不再接收 tuple (input_ids, mask)，直接接收 Tensor
        h_post = x_dict['post']  # Shape: [num_posts, 768]
        
        # 投影到 hidden_channels
        h_post = self.post_proj(h_post)
        
        # 动态初始化 User 投影层 (Lazy Initialization)
        if self.user_proj is None:
            self.user_feat_dim = x_dict['user'].size(1)
            self.user_proj = Linear(self.user_feat_dim, h_post.size(1)).to(h_post.device)
        h_user = self.user_proj(x_dict['user'])
        
        # 动态初始化 Entity 投影层
        if self.entity_proj is None:
            self.entity_feat_dim = x_dict['entity'].size(1)
            self.entity_proj = Linear(self.entity_feat_dim, h_post.size(1)).to(h_post.device)
        h_entity = self.entity_proj(x_dict['entity'])
        
        h_dict = {'post': h_post, 'user': h_user, 'entity': h_entity}
        
        # ========== 2. 异构图卷积 ==========
        for conv in self.convs:
            h_prev = h_dict
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {key: h.relu() for key, h in h_dict.items()}
            # 残差连接/保留未更新节点
            for k in h_prev:
                if k not in h_dict:
                    h_dict[k] = h_prev[k]
        
        # ========== 3. Post 节点分类 ==========
        out = self.classifier(h_dict['post'])
        return out.squeeze(-1)

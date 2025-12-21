# models.py

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATConv, Linear
from transformers import AutoModel

class FraudDetector(nn.Module):
    def __init__(self, text_model_name, hidden_channels, out_channels, metadata, user_feat_dim=None, entity_feat_dim=None):
        super(FraudDetector, self).__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_embed_dim = self.text_encoder.config.hidden_size
        self.user_feat_dim = user_feat_dim
        self.entity_feat_dim = entity_feat_dim
        self.post_proj = Linear(text_embed_dim, hidden_channels)
        self.user_proj = None
        self.entity_proj = None
        self.convs = nn.ModuleList()
        for _ in range(2):
            conv_dict = {
                ('user', 'publish', 'post'): GATConv((-1, -1), hidden_channels, heads=2, concat=False, add_self_loops=False),
                ('user', 'repost', 'post'): GATConv((-1, -1), hidden_channels, heads=2, concat=False, add_self_loops=False),
                ('post', 'contain', 'entity'): GATConv((-1, -1), hidden_channels, heads=2, concat=False, add_self_loops=False), # 修正: mention -> contain
            }
            
            conv_dict[('user', 'interact', 'user')] = GATConv((-1, -1), hidden_channels, heads=2, concat=False, add_self_loops=False)
            conv_dict[('user', 'follow', 'user')] = GATConv((-1, -1), hidden_channels, heads=2, concat=False, add_self_loops=False)
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
        input_ids, attention_mask = x_dict['post']
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        h_post = text_output.last_hidden_state[:, 0, :]
        h_post = self.post_proj(h_post)
        
        if self.user_proj is None:
            self.user_feat_dim = x_dict['user'].size(1)
            self.user_proj = Linear(self.user_feat_dim, h_post.size(1)).to(x_dict['user'].device)
        h_user = self.user_proj(x_dict['user'])
        
        if self.entity_proj is None:
            self.entity_feat_dim = x_dict['entity'].size(1)
            self.entity_proj = Linear(self.entity_feat_dim, h_post.size(1)).to(x_dict['entity'].device)
        h_entity = self.entity_proj(x_dict['entity'])
        
        h_dict = {'post': h_post, 'user': h_user, 'entity': h_entity}
        
        # ========== 2. 异构图卷积 (修复 Crash 的关键部分) ==========
        for conv in self.convs:
            h_prev = h_dict
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {key: h.relu() for key, h in h_dict.items()}
            for k in h_prev:
                if k not in h_dict:
                    h_dict[k] = h_prev[k]  # 直接保留上一层的特征
        
        # ========== 3. Post 节点分类 ==========
        out = self.classifier(h_dict['post'])
        return out.squeeze(-1)

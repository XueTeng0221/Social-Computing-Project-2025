# models.py

import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear
from typing import Tuple, List


class FraudDetector(nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int, 
                 metadata: Tuple[List[str], List[Tuple[str, str, str]]], 
                 text_embed_dim: int = 768, 
                 post_meta_dim: int = 5,
                 cascade_dim: int = 4,  # 新增：级联特征维度
                 num_heads: int = 4, 
                 num_layers: int = 2):    
        """
        参数:
        - hidden_channels: GNN 隐藏层维度
        - out_channels: 输出类别数 (通常为 2)
        - metadata: 异构图的元数据 (node_types, edge_types)
        - text_embed_dim: BERT 文本向量维度 (768)
        - post_meta_dim: 手工特征维度 (关键词, 实体, 长度, 紧急词, 时间戳) -> 共 5 维
        - cascade_dim: 级联特征维度 (默认为4)
        - num_heads: GAT 注意力头数
        - num_layers: GNN 层数
        """
        super(FraudDetector, self).__init__()

        # 1. Post 投影层修改
        # 输入维度 = BERT向量 (768) + Meta特征 (5) + 级联特征 (4) = 777 dims
        # 将融合后的特征映射到 GNN 的 hidden_channels
        self.post_proj = Linear(text_embed_dim + post_meta_dim + cascade_dim, hidden_channels)
        self.user_proj = None
        self.entity_proj = None
        self.media_proj = None  # 如果有 media 节点
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                metadata=metadata,
                heads=num_heads,
                # group='sum'
            )
            self.convs.append(conv)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x_dict, edge_index_dict, post_meta=None, cascade_features=None):
        """
        前向传播
        参数:
        - x_dict: 节点特征字典 (包含 post, user, entity 等的 .x)
        - edge_index_dict: 边索引字典
        - post_meta: [可选] Post 节点的额外统计特征 (Tensor shape: [num_posts, 5])
        - cascade_features: [可选] Post 节点的级联特征 (Tensor shape: [num_posts, 4])
        """
        # ========== 1. 节点特征提取与融合 ==========

        h_post = x_dict['post']  # Shape: [num_posts, 768]
        features_to_concat = [h_post]

        if post_meta is not None:
            if post_meta.device != h_post.device:
                post_meta = post_meta.to(h_post.device)
            features_to_concat.append(post_meta)  # Shape: [num_posts, 5]
            
        if cascade_features is not None:
            if cascade_features.device != h_post.device:
                cascade_features = cascade_features.to(h_post.device)
            features_to_concat.append(cascade_features)  # Shape: [num_posts, 4]

        # 投影融合后的向量 -> hidden_channels (777 dims)
        h_post = self.post_proj(torch.cat(features_to_concat, dim=1))

        # 动态初始化 User 投影层
        if self.user_proj is None:
            self.user_feat_dim = x_dict['user'].size(1)
            self.user_proj = Linear(
                self.user_feat_dim, h_post.size(1)).to(h_post.device)
        h_user = self.user_proj(x_dict['user'])

        # 动态初始化 Entity 投影层
        if 'entity' in x_dict:
            if self.entity_proj is None:
                self.entity_feat_dim = x_dict['entity'].size(1)
                self.entity_proj = Linear(
                    self.entity_feat_dim, h_post.size(1)).to(h_post.device)
            h_entity = self.entity_proj(x_dict['entity'])
        else:
            h_entity = None  # 处理无 entity 的情况

        # 构建当前层的特征字典
        h_dict = {'post': h_post, 'user': h_user}
        if h_entity is not None:
            h_dict['entity'] = h_entity

        # 如果有 domain 节点且在图中，也需要投影，这里略过以保持代码简洁，
        # 实际项目中应遍历 x_dict keys 动态投影

        # ========== 2. 异构图卷积 ==========
        for conv in self.convs:
            h_prev = h_dict.copy()
            h_dict = conv(h_dict, edge_index_dict)
            
            # 残差连接 + 激活
            for key in h_dict:
                if key in h_prev and h_dict[key].shape == h_prev[key].shape:
                    h_dict[key] = h_dict[key] + h_prev[key]
                h_dict[key] = h_dict[key].relu()

        # ========== 3. Post 节点分类 ==========
        # 取出最终的 Post 节点特征进行分类
        out = self.classifier(h_dict['post'])

        return out.squeeze(-1)

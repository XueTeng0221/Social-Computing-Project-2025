# models.py

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATConv, Linear

class FraudDetector(nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata, text_embed_dim=768, post_meta_dim=5):
        """
        参数:
        - hidden_channels: GNN 隐藏层维度
        - out_channels: 输出类别数 (通常为 2)
        - metadata: 异构图的元数据 (node_types, edge_types)
        - text_embed_dim: BERT 文本向量维度 (768)
        - post_meta_dim: 手工特征维度 (关键词, 实体, 长度, 紧急词, 时间戳) -> 共 5 维
        """
        super(FraudDetector, self).__init__()
        
        # 1. Post 投影层修改
        # 输入维度 = BERT向量 (768) + Meta特征 (5)
        # 将融合后的特征映射到 GNN 的 hidden_channels
        self.post_proj = Linear(text_embed_dim + post_meta_dim, hidden_channels)
        
        # 其他节点投影层 (保持懒加载或手动初始化)
        self.user_proj = None
        self.entity_proj = None
        self.media_proj = None # 如果有 media 节点
        
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
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, out_channels)
        )
    
    def forward(self, x_dict, edge_index_dict, post_meta=None):
        """
        前向传播
        参数:
        - x_dict: 节点特征字典 (包含 post, user, entity 等的 .x)
        - edge_index_dict: 边索引字典
        - post_meta: [可选] Post 节点的额外统计特征 (Tensor shape: [num_posts, 5])
        """
        # ========== 1. 节点特征提取与融合 ==========
        
        h_post = x_dict['post']  # Shape: [num_posts, 768]
        
        # [修改] 特征融合逻辑
        if post_meta is not None:
            # 确保 post_meta 在正确的设备上
            if post_meta.device != h_post.device:
                post_meta = post_meta.to(h_post.device)
            # 拼接: [num_posts, 768] + [num_posts, 5] -> [num_posts, 773]
            h_post = torch.cat([h_post, post_meta], dim=1)
        
        # 投影融合后的向量 -> hidden_channels
        h_post = self.post_proj(h_post)
        
        # 动态初始化 User 投影层
        if self.user_proj is None:
            self.user_feat_dim = x_dict['user'].size(1)
            self.user_proj = Linear(self.user_feat_dim, h_post.size(1)).to(h_post.device)
        h_user = self.user_proj(x_dict['user'])
        
        # 动态初始化 Entity 投影层
        if 'entity' in x_dict:
            if self.entity_proj is None:
                self.entity_feat_dim = x_dict['entity'].size(1)
                self.entity_proj = Linear(self.entity_feat_dim, h_post.size(1)).to(h_post.device)
            h_entity = self.entity_proj(x_dict['entity'])
        else:
            h_entity = None # 处理无 entity 的情况
            
        # 构建当前层的特征字典
        h_dict = {'post': h_post, 'user': h_user}
        if h_entity is not None:
            h_dict['entity'] = h_entity
        
        # 如果有 domain 节点且在图中，也需要投影，这里略过以保持代码简洁，
        # 实际项目中应遍历 x_dict keys 动态投影
        
        # ========== 2. 异构图卷积 ==========
        for conv in self.convs:
            h_prev = h_dict
            h_dict = conv(h_dict, edge_index_dict)
            
            # 激活函数
            h_dict = {key: h.relu() for key, h in h_dict.items()}
            
            # 残差连接 (Residual Connection) - 重要，防止深层过平滑
            for k in h_prev:
                if k in h_dict:
                    # 只有维度匹配才能相加，GAT 输出通常维度一致
                    if h_dict[k].shape == h_prev[k].shape:
                        h_dict[k] = h_dict[k] + h_prev[k]
                else:
                    # 对于这一层没有被卷积更新的节点（例如孤立节点），保留上一层特征
                    h_dict[k] = h_prev[k]
        
        # ========== 3. Post 节点分类 ==========
        # 取出最终的 Post 节点特征进行分类
        out = self.classifier(h_dict['post'])
        
        return out.squeeze(-1)

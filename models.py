import torch.nn as nn
from torch_geometric.nn import HANConv
from transformers import AutoModel


class FraudDetector(nn.Module):
    def __init__(self, text_model_name, hidden_channels, out_channels, metadata):
        super().__init__()

        # 1. 文本编码器 (RoBERTa)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        # 冻结部分层以加快训练 (可选)
        for param in self.text_encoder.base_model.parameters():
            param.requires_grad = False
        self.text_proj = nn.Linear(768, hidden_channels)

        # 2. 用户/实体特征投影
        self.user_proj = nn.Linear(16, hidden_channels)  # 假设用户特征16维
        self.entity_proj = nn.Linear(16, hidden_channels)  # 假设实体特征16维 (若用one-hot需调整)

        # 3. 异构图神经网络 (HAN)
        # metadata 是 PyG HeteroData.metadata()
        # 这里定义如何聚合不同类型的边信息
        self.gnn = HANConv(in_channels=hidden_channels, out_channels=hidden_channels,
                           metadata=metadata, heads=4, dropout=0.2)

        # 4. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x_dict, edge_index_dict, batch_dict=None):
        # x_dict: {'post': [input_ids, mask], 'user': features, 'entity': features}

        # A. 编码文本节点
        post_input_ids, post_mask = x_dict['post']
        # 显存优化：如果Post太多，可以使用 mini-batch 或只取 CLS token
        text_out = self.text_encoder(input_ids=post_input_ids, attention_mask=post_mask).last_hidden_state[:, 0,
                   :]  # [CLS]
        h_post = self.text_proj(text_out)

        # B. 编码其他节点
        h_user = self.user_proj(x_dict['user'])
        h_entity = self.entity_proj(x_dict['entity'])

        # 构造 GNN 输入字典
        h_dict = {
            'post': h_post,
            'user': h_user,
            'entity': h_entity
        }

        # C. 图传播 (Message Passing)
        # HAN 会自动处理元路径聚合
        h_gnn = self.gnn(h_dict, edge_index_dict)

        # D. 最终分类 (针对 Post 节点分类)
        # 取出更新后的 post 节点特征
        target_node_feat = h_gnn['post']

        return self.classifier(target_node_feat)

# trainer.py

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.data import HeteroData
from models import FraudDetector


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        if targets.shape != inputs.shape:
            targets = targets.view_as(inputs)
            
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return loss.mean()


def train_epoch(model: FraudDetector, data: HeteroData, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module):
    model.train()
    optimizer.zero_grad()
    x_dict = {
        'post': (data['post'].x, data['post'].mask),
        'user': data['user'].x,
        'entity': data['entity'].x
    }

    out = model(x_dict, data.edge_index_dict)
    mask = data['post'].train_mask
    pred = out[mask].squeeze()
    label = data['post'].y[mask]
    loss = criterion(pred, label.float())
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model: FraudDetector, data: HeteroData, mask: torch.Tensor):
    model.eval()
    with torch.no_grad():
        x_dict = {
            'post': (data['post'].x, data['post'].mask),
            'user': data['user'].x,
            'entity': data['entity'].x
        }
        out = model(x_dict, data.edge_index_dict)
        pred_prob = torch.sigmoid(out[mask]).squeeze().cpu().numpy()
        labels = data['post'].y[mask].cpu().numpy()
        pred_label = (pred_prob > 0.5).astype(int)
        if len(labels) == 0:
            return 0.0, 0.0  # 避免空标签导致的错误
        
        f1 = f1_score(labels, pred_label)
        try:
            auc = roc_auc_score(labels, pred_prob)
        except ValueError:
            # 当测试集中只有一个类别时（例如全是负样本），AUC无法计算
            auc = 0.5 

    return f1, auc

# trainer.py

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.data import HeteroData
from models import FraudDetector

# FocalLoss 保持不变...
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
    
    # 修改点：直接传入 Tensor，去掉 .mask
    x_dict = {
        'post': data['post'].x, 
        'user': data['user'].x,
        'entity': data['entity'].x
    }

    out = model(x_dict, data.edge_index_dict)
    
    # 获取训练集的掩码
    # 注意：确保你的 data['post'] 里有 train_mask
    mask = data['post'].train_mask
    if mask is None:
        raise ValueError("data['post'].train_mask is None. Please split dataset first.")

    pred = out[mask]
    label = data['post'].y[mask]
    
    loss = criterion(pred, label.float())
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model: FraudDetector, data: HeteroData, mask: torch.Tensor):
    model.eval()
    with torch.no_grad():
        # 修改点：同样直接传入 Tensor
        x_dict = {
            'post': data['post'].x,
            'user': data['user'].x,
            'entity': data['entity'].x
        }
        out = model(x_dict, data.edge_index_dict)
        pred_prob = torch.sigmoid(out[mask]).squeeze().cpu().numpy()
        labels = data['post'].y[mask].cpu().numpy()
        
        if len(labels) == 0:
            return 0.0, 0.5
            
        pred_label = (pred_prob > 0.5).astype(int)
        
        f1 = f1_score(labels, pred_label)
        try:
            auc = roc_auc_score(labels, pred_prob)
        except ValueError:
            auc = 0.5 

    return f1, auc
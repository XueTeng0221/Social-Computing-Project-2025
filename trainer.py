# trainer.py

import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from torch_geometric.data import HeteroData
from models import FraudDetector
from typing import Dict, Tuple, List

class WeightedFocalLoss(torch.nn.Module):
    """
    修正后的 Focal Loss，支持对正样本加权
    alpha: 正样本的权重系数 (0 < alpha < 1)，通常设为 0.75 或更高以通过权重解决不平衡
    gamma: 聚焦参数，关注难分样本
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.shape != inputs.shape:
            targets = targets.view_as(inputs)
            
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return loss.mean()

def train_epoch(model: FraudDetector, data: HeteroData, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module) -> float:
    model.train()
    optimizer.zero_grad()
    x_dict = {
        'post': data['post'].x, 
        'user': data['user'].x,
        'entity': data['entity'].x
    }

    out = model(x_dict, data.edge_index_dict, post_meta=data['post'].meta, cascade_features=data['post'].cascade)
    mask = data['post'].train_mask
    pred = out[mask]
    label = data['post'].y[mask]
    loss = criterion(pred, label.float())
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model: FraudDetector, data: HeteroData, mask: torch.Tensor) -> Tuple[float, float, float, float]:
    model.eval()
    with torch.no_grad():
        x_dict = {
            'post': data['post'].x,
            'user': data['user'].x,
            'entity': data['entity'].x
        }
        out = model(x_dict, data.edge_index_dict, post_meta=data['post'].meta, cascade_features=data['post'].cascade)
        pred_prob = torch.sigmoid(out[mask]).squeeze().cpu().numpy()
        labels = data['post'].y[mask].cpu().numpy()
        
        if len(labels) == 0:
            return 0.0, 0.5

        if np.random.rand() < 0.05: # 只有5%的概率打印，避免刷屏
            print(f"\n[Debug] Pred Stats: Mean={pred_prob.mean():.4f}, Max={pred_prob.max():.4f}, Min={pred_prob.min():.4f}")
            print(f"[Debug] True Positives in batch: {labels.sum()}")

        threshold = 0.5
        pred_label = (pred_prob > threshold).astype(int)
        f1 = f1_score(labels, pred_label, zero_division=0)
        precision = precision_score(labels, pred_label, zero_division=0)
        recall = recall_score(labels, pred_label, zero_division=0)
        try:
            auc = roc_auc_score(labels, pred_prob)
        except ValueError:
            auc = 0.5 

    return f1, auc, precision, recall
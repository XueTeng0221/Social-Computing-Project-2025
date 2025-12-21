import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return loss.mean()


def train_epoch(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()

    # 拆解数据
    x_dict = {
        'post': (data['post'].x, data['post'].mask),
        'user': data['user'].x,
        'entity': data['entity'].x
    }

    out = model(x_dict, data.edge_index_dict)

    # 只计算有标签的 Post 节点的 Loss (Masking)
    # 假设我们有 train_mask
    mask = data['post'].train_mask
    pred = out[mask].squeeze()
    label = data['post'].y[mask]

    loss = criterion(pred, label)
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, data, mask):
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

        # 指标计算
        pred_label = (pred_prob > 0.5).astype(int)
        f1 = f1_score(labels, pred_label)
        auc = roc_auc_score(labels, pred_prob)

    return f1, auc

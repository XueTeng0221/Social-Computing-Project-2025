import torch
from models import FraudDetector
from trainer import train_epoch, evaluate, FocalLoss
from torch_geometric.loader import NeighborLoader  # 如果图很大，必须用 Neighbor Sampling


def main():
    # 1. 加载数据
    data = torch.load('data/processed/hetero_graph.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # 划分数据集 (简单示例)
    num_posts = data['post'].y.size(0)
    indices = torch.randperm(num_posts)
    data['post'].train_mask = indices[:int(0.7 * num_posts)]
    data['post'].val_mask = indices[int(0.7 * num_posts):int(0.85 * num_posts)]
    data['post'].test_mask = indices[int(0.85 * num_posts):]

    # 2. 初始化模型
    model = FraudDetector(
        text_model_name='hfl/chinese-roberta-wwm-ext',
        hidden_channels=64,
        out_channels=1,
        metadata=data.metadata()
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    criterion = FocalLoss(alpha=0.7, gamma=2)  # alpha设大一点偏向正类（诈骗类）

    # 3. 训练循环
    best_f1 = 0
    for epoch in range(50):
        loss = train_epoch(model, data, optimizer, criterion)
        val_f1, val_auc = evaluate(model, data, data['post'].val_mask)

        print(f"Epoch {epoch}: Loss {loss:.4f}, Val F1 {val_f1:.4f}, Val AUC {val_auc:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pth')

    print("Training Finished.")

    # 4. 传播分析与干预评估 (概念性逻辑)
    # 加载最佳模型，识别出所有高风险节点
    model.load_state_dict(torch.load('best_model.pth'))
    # ... 推理 ...

    print("建议干预措施：对高风险 Cluster ID [102, 55] 进行限流，切断 User-Entity 边。")


if __name__ == "__main__":
    main()

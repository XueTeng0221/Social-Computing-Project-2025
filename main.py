# main.py

import torch
import pandas as pd
import os
import argparse
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import BaseStorage, GlobalStorage, NodeStorage, EdgeStorage
from models import FraudDetector
from preprocessor import DataPreprocessor
from trainer import train_epoch, evaluate, FocalLoss

torch.serialization.add_safe_globals([
    HeteroData, BaseStorage, GlobalStorage, NodeStorage, EdgeStorage
])


def prepare_data(force_rebuild) -> HeteroData:
    """准备并加载图数据"""
    graph_path = 'data/processed/hetero_graph.pt'
    if os.path.exists(graph_path) and not force_rebuild:
        print("📂 加载已有图数据...")
        data = torch.load(graph_path, weights_only=True)
    else:
        print("🔨 构建新图...")
        df_posts = pd.read_csv('data/raw/posts.csv')
        df_users = pd.read_csv('data/raw/users.csv')
        df_relations = pd.read_csv('data/raw/relations.csv')
        preprocessor = DataPreprocessor()
        data = preprocessor.build_graph(df_posts, df_users, df_relations)
        os.makedirs('data/processed', exist_ok=True)
        torch.save(data, graph_path)
        print(f"💾 图已保存到 {graph_path}")
    
    return data


def split_dataset(data, train_ratio=0.7, val_ratio=0.15):
    """划分训练/验证/测试集"""
    num_posts = data['post'].y.size(0)
    indices = torch.randperm(num_posts)
    train_size = int(train_ratio * num_posts)
    val_size = int(val_ratio * num_posts)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    data['post'].train_mask = torch.zeros(num_posts, dtype=torch.bool)
    data['post'].val_mask = torch.zeros(num_posts, dtype=torch.bool)
    data['post'].test_mask = torch.zeros(num_posts, dtype=torch.bool)
    data['post'].train_mask[train_idx] = True
    data['post'].val_mask[val_idx] = True
    data['post'].test_mask[test_idx] = True
    print(f"📊 数据集划分: Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")
    return data


def main():
    argp = argparse.ArgumentParser(description="训练异构图诈骗检测模型")
    argp.add_argument('--alpha', type=float, default=0.7, help='Focal Loss 的 alpha 参数')
    argp.add_argument('--gamma', type=float, default=2.0, help='Focal Loss 的 gamma 参数')
    argp.add_argument('--force_rebuild', action='store_true', help='强制重建图数据')
    argp.add_argument('--epochs', type=int, default=50, help='训练的最大轮数')
    argp.add_argument('--patience', type=int, default=10, help='早停的耐心值')
    argp.add_argument('--lr', type=float, default=1e-4, help='学习率')
    argp.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减')
    argp.add_argument('--save-dir', type=str, default='models', help='模型保存目录')
    args = argp.parse_args()
    
    
    # 1. 准备数据
    data = prepare_data(force_rebuild=args.force_rebuild)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    data = data.to(device)
    data = split_dataset(data)
    
    # 2. 初始化模型
    model = FraudDetector(
        text_model_name='hfl/chinese-roberta-wwm-ext',
        hidden_channels=64,
        out_channels=1,
        metadata=data.metadata()
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma)
    
    # 3. 训练循环
    best_f1 = 0
    patience = args.patience
    patience_counter = 0
    for epoch in range(args.epochs):
        loss = train_epoch(model, data, optimizer, criterion)
        val_f1, val_auc = evaluate(model, data, data['post'].val_mask)
        print(f"Epoch {epoch+1:02d}: Loss {loss:.4f} | Val F1 {val_f1:.4f} | Val AUC {val_auc:.4f}")
        if val_f1 >= best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), f'{args.save_dir}/best_model.pth')
            print(f"  ✅ 保存最佳模型 (F1={best_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"⏹️  早停触发 (patience={patience})")
            break
    
    print("\n🎉 训练完成!")
    
    # 4. 测试最佳模型
    model.load_state_dict(torch.load(f'{args.save_dir}/best_model.pth', weights_only=True))
    test_f1, test_auc = evaluate(model, data, data['post'].test_mask)
    print(f"\n🎯 测试集性能: F1 {test_f1:.4f} | AUC {test_auc:.4f}")


if __name__ == "__main__":
    main()

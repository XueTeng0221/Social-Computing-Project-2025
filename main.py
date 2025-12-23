import torch
import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.nn.parameter import UninitializedParameter
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import BaseStorage, GlobalStorage, NodeStorage, EdgeStorage
from models import FraudDetector
from preprocessor import DataPreprocessor
from sklearn.preprocessing import StandardScaler
from trainer import train_epoch, evaluate, WeightedFocalLoss

# --- 新增：可视化工具函数 ---
def plot_training_history(history, save_dir):
    """绘制训练过程中的 Loss 和 指标变化"""
    epochs = range(1, len(history['loss']) + 1)
    
    plt.figure(figsize=(15, 6))
    
    # 子图 1: Loss 变化
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], 'b-', label='Training Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # 子图 2: 验证集指标变化
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_f1'], 'r-', label='Val F1')
    plt.plot(epochs, history['val_auc'], 'g--', label='Val AUC')
    plt.plot(epochs, history['val_precision'], 'c:', label='Val Precision')
    plt.plot(epochs, history['val_recall'], 'm:', label='Val Recall')
    plt.title('Validation Metrics per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'training_history_alpha={args.alpha}_gamma={args.gamma}.png')
    plt.savefig(save_path, dpi=300)
    print(f"📊 训练历史图表已保存至: {save_path}")

def plot_confusion_matrix_result(model, data, mask, save_dir, title="Test Confusion Matrix"):
    """绘制混淆矩阵"""
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict, post_meta=data['post'].meta)
        # 获取预测结果
        pred = (out[mask] > 0).float().cpu().numpy()
        y_true = data['post'].y[mask].cpu().numpy()
    
    cm = confusion_matrix(y_true, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Fraud'])
    
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(title)
    
    save_path = os.path.join(save_dir, f'confusion_matrix_alpha={args.alpha}_gamma={args.gamma}.png')
    plt.savefig(save_path, dpi=300)
    print(f"📊 混淆矩阵已保存至: {save_path}")

def inspect_label_distribution(data):
    """检查标签分布，计算建议的 alpha 值"""
    y = data['post'].y
    num_pos = y.sum().item()
    num_total = y.size(0)
    num_neg = num_total - num_pos
    
    print(f"\n📊 标签分布统计:")
    print(f"  - 总样本数: {num_total}")
    print(f"  - 欺诈样本 (Label=1): {num_pos} ({num_pos/num_total:.2%})")
    print(f"  - 正常样本 (Label=0): {num_neg} ({num_neg/num_total:.2%})")
    
    if num_pos == 0:
        raise ValueError("❌ 数据集中没有正样本（欺诈样本）！模型无法训练。")

    suggested_alpha = num_neg / num_total
    print(f"💡 建议 Focal Loss Alpha: {suggested_alpha:.4f}")
    return suggested_alpha

torch.serialization.add_safe_globals([
    HeteroData, BaseStorage, GlobalStorage, NodeStorage, EdgeStorage, UninitializedParameter
])

def check_and_normalize_data(data: HeteroData):
    print("\n🔍 正在检查数据质量...")
    has_nan = False
    for node_type in data.node_types:
        if torch.isnan(data[node_type].x).any():
            has_nan = True
        if torch.isinf(data[node_type].x).any():
            has_nan = True
            
    if has_nan:
        print("⚠️ 检测到异常数值，尝试将其替换为 0...")
        for node_type in data.node_types:
            data[node_type].x = torch.nan_to_num(data[node_type].x, nan=0.0, posinf=1.0, neginf=-1.0)

    if 'user' in data.node_types:
        print("⚖️ 对 User 特征进行归一化...")
        scaler = StandardScaler()
        user_x = data['user'].x.cpu().numpy()
        user_x = scaler.fit_transform(user_x)
        data['user'].x = torch.tensor(user_x, dtype=torch.float32)
        
    print("✅ 数据检查与预处理完成")
    return data

def prepare_data(force_rebuild) -> HeteroData:
    graph_path = 'data/processed/hetero_graph.pt'
    if not force_rebuild and os.path.exists(graph_path):
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

if __name__ == "__main__":
    argp = argparse.ArgumentParser(description="训练异构图诈骗检测模型")
    argp.add_argument('--alpha', type=float, default=0.7, help='Focal Loss 的 alpha 参数')
    argp.add_argument('--gamma', type=float, default=2.0, help='Focal Loss 的 gamma 参数')
    argp.add_argument('--force-rebuild', action='store_true', help='强制重建图数据') # 修正了 bool 参数的写法
    argp.add_argument('--epochs', type=int, default=50, help='训练的最大轮数')
    argp.add_argument('--lr', type=float, default=1e-4, help='学习率')
    argp.add_argument('--weight-decay', type=float, default=5e-4, help='权重衰减')
    argp.add_argument('--save-dir', type=str, default='models', help='模型保存目录')
    args = argp.parse_args()
    
    # 确保保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)

    data = prepare_data(force_rebuild=args.force_rebuild)
    data = check_and_normalize_data(data)
    suggested_alpha = inspect_label_distribution(data)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    data = data.to(device)
    data = split_dataset(data)
    
    model = FraudDetector(
        hidden_channels=64,
        out_channels=1,
        metadata=data.metadata()
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = WeightedFocalLoss(alpha=args.alpha, gamma=args.gamma)
    
    print(f"🚀 模型初始化完成")
    
    # --- 初始化历史记录字典 ---
    history = {
        'loss': [],
        'val_f1': [],
        'val_auc': [],
        'val_precision': [],
        'val_recall': []
    }

    best_f1 = 0
    for epoch in range(args.epochs):
        loss = train_epoch(model, data, optimizer, criterion)
        val_f1, val_auc, precision, recall = evaluate(model, data, data['post'].val_mask)
        
        # --- 记录数据 ---
        history['loss'].append(loss)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)

        print(f"Epoch {epoch+1:02d}: Loss {loss:.4f} | Val F1 {val_f1:.4f} | Val AUC {val_auc:.4f}")
        
        if val_f1 >= best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), f'{args.save_dir}/best_model_alpha={args.alpha}_gamma={args.gamma}.pth')
            print(f"  ✅ 保存最佳模型 (F1={best_f1:.4f})")
    
    print("\n🎉 训练完成!")
    
    # --- 绘图：训练曲线 ---
    print("\n🎨 正在绘制训练曲线...")
    plot_training_history(history, args.save_dir)

    # 4. 测试最佳模型
    print("\n🔍 加载最佳模型进行测试...")
    model.load_state_dict(torch.load(f'{args.save_dir}/best_model_alpha={args.alpha}_gamma={args.gamma}.pth', weights_only=True))
    test_f1, test_auc, test_precision, test_recall = evaluate(model, data, data['post'].test_mask)
    print(f"🎯 测试集性能: F1 {test_f1:.4f} | AUC {test_auc:.4f} | Precision {test_precision:.4f} | Recall {test_recall:.4f}")
    
    # --- 绘图：混淆矩阵 ---
    print("\n🎨 正在绘制混淆矩阵...")
    plot_confusion_matrix_result(model, data, data['post'].test_mask, args.save_dir)

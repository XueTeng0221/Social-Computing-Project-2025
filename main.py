import torch
import pandas as pd
import os
import argparse
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from torch.nn.parameter import UninitializedParameter
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import BaseStorage, GlobalStorage, NodeStorage, EdgeStorage
from models import FraudDetector
from preprocessor import DataPreprocessor
from sklearn.preprocessing import StandardScaler
from trainer import train_epoch, evaluate, WeightedFocalLoss

def plot_comprehensive_results(history, model, data, test_mask, save_dir, alpha, gamma):
    """
    综合绘制 4 个子图：
    ① Loss 曲线
    ② 验证集指标曲线
    ③ 测试集混淆矩阵
    ④ 测试集 ROC 曲线
    """
    fig = plt.figure(figsize=(16, 12))
    
    # ========== 子图 1: Loss 曲线 ==========
    ax1 = plt.subplot(2, 2, 1)
    epochs = range(1, len(history['loss']) + 1)
    ax1.plot(epochs, history['loss'], 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_title('① Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_facecolor('#f9f9f9')
    
    # ========== 子图 2: 验证集指标曲线 ==========
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(epochs, history['val_f1'], 'r-', linewidth=2, marker='s', markersize=4, label='F1 Score')
    ax2.plot(epochs, history['val_auc'], 'g--', linewidth=2, marker='^', markersize=4, label='AUC')
    ax2.plot(epochs, history['val_precision'], 'c:', linewidth=2, marker='D', markersize=4, label='Precision')
    ax2.plot(epochs, history['val_recall'], 'm-.', linewidth=2, marker='v', markersize=4, label='Recall')
    ax2.set_title('② Validation Metrics', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_facecolor('#f9f9f9')
    
    # ========== 子图 3: 测试集混淆矩阵 ==========
    ax3 = plt.subplot(2, 2, 3)
    model.eval()
    with torch.no_grad():
        x_dict = {
            'post': data['post'].x,
            'user': data['user'].x,
            'entity': data['entity'].x
        }
        out = model(x_dict, data.edge_index_dict, post_meta=data['post'].meta, cascade_features=data['post'].cascade)
        pred_prob = torch.sigmoid(out[test_mask]).squeeze().cpu().numpy()
        pred_label = (pred_prob > 0.5).astype(int)
        y_true = data['post'].y[test_mask].cpu().numpy()
    
    cm = confusion_matrix(y_true, pred_label)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Fraud'])
    disp.plot(ax=ax3, cmap='Blues', values_format='d', colorbar=False)
    ax3.set_title('③ Test Confusion Matrix', fontsize=14, fontweight='bold')
    ax3.grid(False)
    
    # ========== 子图 4: 测试集 ROC 曲线 ==========
    ax4 = plt.subplot(2, 2, 4)
    fpr, tpr, _ = roc_curve(y_true, pred_prob)
    roc_auc = auc(fpr, tpr)
    
    ax4.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax4.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
    ax4.set_title('④ Test ROC Curve', fontsize=14, fontweight='bold')
    ax4.set_xlabel('False Positive Rate', fontsize=12)
    ax4.set_ylabel('True Positive Rate', fontsize=12)
    ax4.legend(loc='lower right', fontsize=10)
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.set_facecolor('#f9f9f9')
    
    # 调整布局并保存
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'comprehensive_results_alpha={alpha}_gamma={gamma}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 综合结果图已保存至: {save_path}")

def save_metrics_to_csv(history, test_metrics, save_dir, alpha, gamma):
    """
    将训练历史和测试指标保存为 CSV 文件
    """
    # 保存训练历史
    history_df = pd.DataFrame({
        'Epoch': range(1, len(history['loss']) + 1),
        'Loss': history['loss'],
        'Val_F1': history['val_f1'],
        'Val_AUC': history['val_auc'],
        'Val_Precision': history['val_precision'],
        'Val_Recall': history['val_recall']
    })
    history_path = os.path.join(save_dir, f'training_history_alpha={alpha}_gamma={gamma}.csv')
    history_df.to_csv(history_path, index=False)
    print(f"📄 训练历史已保存至: {history_path}")
    
    # 保存测试指标
    test_df = pd.DataFrame([test_metrics])
    test_path = os.path.join(save_dir, f'test_metrics_alpha={alpha}_gamma={gamma}.csv')
    test_df.to_csv(test_path, index=False)
    print(f"📄 测试指标已保存至: {test_path}")

def inspect_label_distribution(data):
    """检查标签分布，计算建议的 alpha 值"""
    y = data['post'].y
    num_pos = y.sum().item()
    num_total = y.size(0)
    num_neg = num_total - num_pos
    
    logger.info(f"\n📊 标签分布统计:")
    logger.info(f"  - 总样本数: {num_total}")
    logger.info(f"  - 欺诈样本 (Label=1): {num_pos} ({num_pos/num_total:.2%})")
    logger.info(f"  - 正常样本 (Label=0): {num_neg} ({num_neg/num_total:.2%})")
    
    if num_pos == 0:
        raise ValueError("❌ 数据集中没有正样本（欺诈样本）！模型无法训练。")

    suggested_alpha = num_neg / num_total
    logger.info(f"💡 建议 Focal Loss Alpha: {suggested_alpha:.4f}")
    return suggested_alpha

torch.serialization.add_safe_globals([
    HeteroData, BaseStorage, GlobalStorage, NodeStorage, EdgeStorage, UninitializedParameter
])

def check_and_normalize_data(data: HeteroData):
    logger.info("\n🔍 正在检查数据质量...")
    has_nan = False
    for node_type in data.node_types:
        if torch.isnan(data[node_type].x).any():
            has_nan = True
        if torch.isinf(data[node_type].x).any():
            has_nan = True
            
    if has_nan:
        logger.info("⚠️ 检测到异常数值，尝试将其替换为 0...")
        for node_type in data.node_types:
            data[node_type].x = torch.nan_to_num(data[node_type].x, nan=0.0, posinf=1.0, neginf=-1.0)

    if 'user' in data.node_types:
        logger.info("⚖️ 对 User 特征进行归一化...")
        scaler = StandardScaler()
        user_x = data['user'].x.cpu().numpy()
        user_x = scaler.fit_transform(user_x)
        data['user'].x = torch.tensor(user_x, dtype=torch.float32)
        
    logger.info("✅ 数据检查与预处理完成")
    return data

def prepare_data(force_rebuild) -> HeteroData:
    graph_path = 'data/processed/hetero_graph.pt'
    if not force_rebuild and os.path.exists(graph_path):
        logger.info("📂 加载已有图数据...")
        data = torch.load(graph_path, weights_only=True)
    else:
        logger.info("🔨 构建新图...")
        df_posts = pd.read_csv('data/raw/posts.csv')
        df_users = pd.read_csv('data/raw/users.csv')
        df_relations = pd.read_csv('data/raw/relations.csv')
        preprocessor = DataPreprocessor()
        data = preprocessor.build_graph(df_posts, df_users, df_relations)
        os.makedirs('data/processed', exist_ok=True)
        torch.save(data, graph_path)
        logger.info(f"💾 图已保存到 {graph_path}")
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
    logger.info(f"📊 数据集划分: Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")
    return data

if __name__ == "__main__":
    argp = argparse.ArgumentParser(description="训练异构图诈骗检测模型")
    argp.add_argument('--alpha', type=float, default=0.7, help='Focal Loss 的 alpha 参数')
    argp.add_argument('--gamma', type=float, default=2.0, help='Focal Loss 的 gamma 参数')
    argp.add_argument('--force-rebuild', action='store_true', help='强制重建图数据')
    argp.add_argument('--epochs', type=int, default=50, help='训练的最大轮数')
    argp.add_argument('--lr', type=float, default=1e-4, help='学习率')
    argp.add_argument('--weight-decay', type=float, default=5e-4, help='权重衰减')
    argp.add_argument('--save-dir', type=str, default='results', help='模型保存目录')
    args = argp.parse_args()
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S', filename='training.log', 
                        filemode='a', encoding='utf-8')
    logger = logging.getLogger()
    
    os.makedirs(args.save_dir, exist_ok=True)
    data = prepare_data(force_rebuild=args.force_rebuild)
    data = check_and_normalize_data(data)
    suggested_alpha = inspect_label_distribution(data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  使用设备: {device}")
    data = data.to(device)
    data = split_dataset(data)
    model = FraudDetector(
        hidden_channels=64,
        out_channels=1,
        metadata=data.metadata()
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = WeightedFocalLoss(alpha=args.alpha, gamma=args.gamma)
    logger.info(f"🚀 模型初始化完成")
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
        history['loss'].append(loss)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        logger.info(f"Epoch {epoch+1:02d}: Loss {loss:.4f} | Val F1 {val_f1:.4f} | Val AUC {val_auc:.4f} | Precision {precision:.4f} | Recall {recall:.4f}")
        if val_f1 >= best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), f'models/best_model_alpha={args.alpha}_gamma={args.gamma}.pth')
            logger.info(f"  ✅ 保存最佳模型 (F1={best_f1:.4f})")
    
    logger.info("\n🎉 训练完成!")
    logger.info("\n🔍 加载最佳模型进行测试...")
    model.load_state_dict(torch.load(f'models/best_model_alpha={args.alpha}_gamma={args.gamma}.pth', weights_only=True))
    test_f1, test_auc, test_precision, test_recall = evaluate(model, data, data['post'].test_mask)
    test_metrics = {
        'Test_F1': test_f1,
        'Test_AUC': test_auc,
        'Test_Precision': test_precision,
        'Test_Recall': test_recall
    }
    
    logger.info(f"🎯 测试集性能: F1 {test_f1:.4f} | AUC {test_auc:.4f} | Precision {test_precision:.4f} | Recall {test_recall:.4f}")
    logger.info("\n🎨 正在绘制结果图...")
    plot_comprehensive_results(history, model, data, data['post'].test_mask, args.save_dir, args.alpha, args.gamma)
    print("\n💾 正在保存指标数据...")
    save_metrics_to_csv(history, test_metrics, args.save_dir, args.alpha, args.gamma)
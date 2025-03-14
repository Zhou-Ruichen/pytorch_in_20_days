import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
from matplotlib.ticker import MaxNLocator
import itertools
import os
from scipy import interpolate
from typing import List, Dict, Tuple, Optional, Union

# 设置Matplotlib风格，提高输出质量
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16


def create_output_dir(output_dir: str = 'paper_figures') -> str:
    """
    创建输出目录
    
    Args:
        output_dir: 输出目录路径
    
    Returns:
        创建的目录路径
    """
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_training_history(history: pd.DataFrame, 
                          output_dir: str = 'paper_figures',
                          filename_prefix: str = 'training',
                          figsize: Tuple[int, int] = (12, 10),
                          custom_title: Optional[str] = None) -> None:
    """
    绘制训练历史曲线（损失和准确率）
    
    Args:
        history: 包含训练历史的DataFrame，通常由train_model函数返回
        output_dir: 图像保存目录
        filename_prefix: 文件名前缀
        figsize: 图像尺寸
        custom_title: 自定义标题，默认为None
    """
    create_output_dir(output_dir)
    
    metric_columns = [col for col in history.columns if 'loss' not in col.lower() and 'epoch' not in col.lower()]
    loss_columns = [col for col in history.columns if 'loss' in col.lower()]
    
    # 创建两个子图
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # 绘制损失曲线
    for col in loss_columns:
        axes[0].plot(history.index + 1, history[col], marker='o', linestyle='-', 
                   label=col.replace('_', ' ').title())
    
    axes[0].set_ylabel('Loss')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 绘制指标曲线（如准确率）
    for col in metric_columns:
        axes[1].plot(history.index + 1, history[col], marker='o', linestyle='-', 
                   label=col.replace('_', ' ').title())
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Metric Value')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # 设置主标题
    title = custom_title if custom_title else 'Training and Validation Metrics'
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)  # 为标题留出空间
    
    # 保存图像
    plt.savefig(f"{output_dir}/{filename_prefix}_history.png", bbox_inches='tight')
    plt.savefig(f"{output_dir}/{filename_prefix}_history.pdf", bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(true_labels: np.ndarray, 
                          pred_labels: np.ndarray, 
                          class_names: List[str],
                          output_dir: str = 'paper_figures',
                          filename: str = 'confusion_matrix',
                          figsize: Tuple[int, int] = (10, 8),
                          normalize: bool = True,
                          cmap: str = 'Blues',
                          title: Optional[str] = None) -> None:
    """
    绘制混淆矩阵
    
    Args:
        true_labels: 真实标签数组
        pred_labels: 预测标签数组
        class_names: 类别名称列表
        output_dir: 输出目录
        filename: 输出文件名
        figsize: 图像尺寸
        normalize: 是否归一化混淆矩阵
        cmap: 颜色映射
        title: 图表标题
    """
    create_output_dir(output_dir)
    
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)
    
    # 归一化
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # 绘图
    plt.figure(figsize=figsize)
    if title:
        plt.title(title)
    else:
        plt.title('Confusion Matrix')
    
    # 设置颜色映射
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    # 设置刻度标记
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # 添加文本注释
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 保存图像
    plt.savefig(f"{output_dir}/{filename}.png", bbox_inches='tight')
    plt.savefig(f"{output_dir}/{filename}.pdf", bbox_inches='tight')
    plt.close()


def generate_classification_report(true_labels: np.ndarray, 
                                  pred_labels: np.ndarray,
                                  class_names: List[str],
                                  output_dir: str = 'paper_figures',
                                  filename: str = 'classification_report') -> pd.DataFrame:
    """
    生成分类报告并保存为CSV
    
    Args:
        true_labels: 真实标签数组
        pred_labels: 预测标签数组
        class_names: 类别名称列表
        output_dir: 输出目录
        filename: 输出文件名
    
    Returns:
        包含分类指标的DataFrame
    """
    create_output_dir(output_dir)
    
    # 生成分类报告
    report = classification_report(true_labels, pred_labels, 
                                   target_names=class_names, 
                                   output_dict=True)
    
    # 转换为DataFrame
    df_report = pd.DataFrame(report).transpose()
    
    # 保存为CSV
    df_report.to_csv(f"{output_dir}/{filename}.csv")
    
    # 创建可视化表格
    plt.figure(figsize=(12, 8))
    plt.axis('off')
    
    # 选择要显示的列
    display_cols = ['precision', 'recall', 'f1-score', 'support']
    table_data = df_report[display_cols].iloc[:-3].copy()  # 排除avg行
    
    # 将support列转为整数
    table_data['support'] = table_data['support'].astype(int)
    
    # 格式化为表格
    for col in display_cols[:3]:  # 不包括support列
        table_data[col] = table_data[col].map('{:.3f}'.format)
    
    # 创建表格
    table = plt.table(cellText=table_data.values,
                     rowLabels=table_data.index,
                     colLabels=display_cols,
                     loc='center',
                     cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title('Classification Report', fontsize=16)
    plt.tight_layout()
    
    # 保存表格为图像
    plt.savefig(f"{output_dir}/{filename}_table.png", bbox_inches='tight')
    plt.savefig(f"{output_dir}/{filename}_table.pdf", bbox_inches='tight')
    plt.close()
    
    return df_report


def plot_roc_curves(true_labels: np.ndarray, 
                    pred_probs: np.ndarray,
                    class_names: List[str],
                    output_dir: str = 'paper_figures',
                    filename: str = 'roc_curves',
                    figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    绘制ROC曲线（多分类）
    
    Args:
        true_labels: 真实标签（一维数组）
        pred_probs: 预测概率（二维数组，形状为[样本数, 类别数]）
        class_names: 类别名称列表
        output_dir: 输出目录
        filename: 输出文件名
        figsize: 图像尺寸
    """
    create_output_dir(output_dir)
    
    n_classes = len(class_names)
    
    # 转换为one-hot编码
    y_true_bin = np.eye(n_classes)[true_labels]
    
    # 计算每个类别的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 计算微平均ROC曲线和AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), pred_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # 计算宏平均ROC曲线和AUC
    # 首先对所有FPR点进行插值
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # 然后使用插值来计算在这些点上的平均TPR
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # 最后平均并计算AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # 绘图
    plt.figure(figsize=figsize)
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # 绘制每个类别的ROC曲线
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    # 绘制微平均ROC曲线
    plt.plot(fpr["micro"], tpr["micro"],
            label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
            color='deeppink', linestyle=':', linewidth=4)
    
    # 绘制宏平均ROC曲线
    plt.plot(fpr["macro"], tpr["macro"],
            label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})',
            color='navy', linestyle=':', linewidth=4)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.png", bbox_inches='tight')
    plt.savefig(f"{output_dir}/{filename}.pdf", bbox_inches='tight')
    plt.close()


def plot_precision_recall_curves(true_labels: np.ndarray, 
                                pred_probs: np.ndarray,
                                class_names: List[str],
                                output_dir: str = 'paper_figures',
                                filename: str = 'precision_recall_curves',
                                figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    绘制精确率-召回率曲线（多分类）
    
    Args:
        true_labels: 真实标签（一维数组）
        pred_probs: 预测概率（二维数组，形状为[样本数, 类别数]）
        class_names: 类别名称列表
        output_dir: 输出目录
        filename: 输出文件名
        figsize: 图像尺寸
    """
    create_output_dir(output_dir)
    
    n_classes = len(class_names)
    
    # 转换为one-hot编码
    y_true_bin = np.eye(n_classes)[true_labels]
    
    # 计算每个类别的精确率-召回率曲线
    precision = dict()
    recall = dict()
    avg_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], pred_probs[:, i])
        # 计算平均精确率
        avg_precision[i] = np.mean([p for p, r in zip(precision[i][:-1], recall[i][:-1]) if r > 0])
    
    # 绘图
    plt.figure(figsize=figsize)
    
    # 绘制每个类别的PR曲线
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                label=f'{class_names[i]} (AP = {avg_precision[i]:.3f})')
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.png", bbox_inches='tight')
    plt.savefig(f"{output_dir}/{filename}.pdf", bbox_inches='tight')
    plt.close()


def plot_tsne_features(features, 
                      labels, 
                      class_names, 
                      output_dir='paper_figures',
                      filename='tsne_visualization',
                      figsize=(12, 10),
                      perplexity=30,  # 这是问题所在
                      random_state=42):
    """
    使用t-SNE可视化特征分布
    
    Args:
        features: 特征数组，形状为[样本数, 特征维度]
        labels: 标签数组
        class_names: 类别名称列表
        output_dir: 输出目录
        filename: 输出文件名
        figsize: 图像尺寸
        perplexity: t-SNE的perplexity参数
        random_state: 随机种子
    """
    # 添加这段代码在函数开始处
    n_samples = features.shape[0]
    if perplexity >= n_samples:
        # 自动调整perplexity为样本数的一半或5（取较大值）
        perplexity = max(5, n_samples // 2 - 1)
        print(f"已自动调整perplexity为{perplexity}以适应样本数量{n_samples}")


    create_output_dir(output_dir)
    
    # 应用t-SNE降维
    print("正在计算t-SNE特征嵌入...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    features_embedded = tsne.fit_transform(features)
    
    # 绘制散点图
    plt.figure(figsize=figsize)
    
    # 为每个类别使用不同颜色
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    for i, color in zip(range(len(class_names)), colors):
        idx = labels == i
        plt.scatter(features_embedded[idx, 0], features_embedded[idx, 1], 
                   color=color, s=50, alpha=0.7, 
                   label=class_names[i])
    
    plt.title('t-SNE Visualization of Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.png", bbox_inches='tight')
    plt.savefig(f"{output_dir}/{filename}.pdf", bbox_inches='tight')
    plt.close()


def extract_features_and_predictions(model, dataloader, device, feature_layer_name=None):
    """
    提取特征和预测结果
    
    Args:
        model: PyTorch模型
        dataloader: 数据加载器
        device: 运行设备
        feature_layer_name: 要提取特征的层名称（可选）
    
    Returns:
        features, true_labels, pred_labels, pred_probs
    """
    model.eval()
    
    all_features = []
    all_true_labels = []
    all_pred_labels = []
    all_pred_probs = []
    
    # 注册钩子来获取特定层的输出
    activation = {}
    if feature_layer_name:
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        # 根据层名注册钩子
        for name, module in model.named_modules():
            if name == feature_layer_name:
                module.register_forward_hook(get_activation(feature_layer_name))
                break
    
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            labels = labels.to(device)
            
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            
            probs = torch.softmax(outputs, dim=1)
            
            all_true_labels.append(labels.cpu().numpy())
            all_pred_labels.append(preds.cpu().numpy())
            all_pred_probs.append(probs.cpu().numpy())
            
            # 如果指定了特征层，收集该层的输出
            if feature_layer_name and feature_layer_name in activation:
                features = activation[feature_layer_name]
                
                # 如果特征是四维的（如卷积层输出），展平为二维
                if len(features.shape) == 4:  # [B, C, H, W]
                    features = features.mean([2, 3])  # 全局平均池化
                
                all_features.append(features.cpu().numpy())
    
    # 连接所有批次的结果
    true_labels = np.concatenate(all_true_labels)
    pred_labels = np.concatenate(all_pred_labels)
    pred_probs = np.concatenate(all_pred_probs)
    
    features = None
    if all_features:
        features = np.concatenate(all_features)
    
    return features, true_labels, pred_labels, pred_probs


def generate_evaluation_plots(model, val_loader, class_names, history_df=None, 
                             feature_layer_name=None, output_dir='paper_figures',
                             device=None):
    """
    生成所有评估图表的主函数
    
    Args:
        model: PyTorch模型
        val_loader: 验证数据加载器
        class_names: 类别名称列表
        history_df: 训练历史DataFrame（可选）
        feature_layer_name: 要提取特征的层名称（可选）
        output_dir: 输出目录
        device: 运行设备
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"正在使用设备: {device}")
    model = model.to(device)
    
    # 提取特征和预测结果
    features, true_labels, pred_labels, pred_probs = extract_features_and_predictions(
        model, val_loader, device, feature_layer_name)
    
    # 创建输出目录
    create_output_dir(output_dir)
    
    # 1. 绘制训练历史曲线
    if history_df is not None:
        print("绘制训练历史曲线...")
        plot_training_history(history_df, output_dir)
    
    # 2. 绘制混淆矩阵
    print("绘制混淆矩阵...")
    plot_confusion_matrix(true_labels, pred_labels, class_names, output_dir)
    
    # 3. 生成分类报告
    print("生成分类报告...")
    report_df = generate_classification_report(true_labels, pred_labels, class_names, output_dir)
    
    # 4. 绘制ROC曲线
    print("绘制ROC曲线...")
    plot_roc_curves(true_labels, pred_probs, class_names, output_dir)
    
    # 5. 绘制精确率-召回率曲线
    print("绘制精确率-召回率曲线...")
    plot_precision_recall_curves(true_labels, pred_probs, class_names, output_dir)
    
    # 6. 如果有特征数据，绘制t-SNE可视化
    if features is not None:
        print("正在绘制t-SNE特征可视化...")
        plot_tsne_features(features, true_labels, class_names, output_dir)
    
    print(f"所有图表已保存到目录: {output_dir}")
    
    # 返回评估指标
    return {
        "classification_report": report_df,
        "accuracy": (true_labels == pred_labels).mean()
    }


def plot_model_comparison(models_metrics: Dict[str, float], 
                        metric_name: str = 'Accuracy',
                        output_dir: str = 'paper_figures',
                        filename: str = 'model_comparison',
                        figsize: Tuple[int, int] = (12, 8),
                        title: Optional[str] = None,
                        sort_values: bool = True):
    """
    绘制模型比较条形图
    
    Args:
        models_metrics: 包含模型名称和对应指标值的字典
        metric_name: 指标名称
        output_dir: 输出目录
        filename: 输出文件名
        figsize: 图像尺寸
        title: 图表标题
        sort_values: 是否按指标值排序
    """
    create_output_dir(output_dir)
    
    # 创建DataFrame
    df = pd.DataFrame(list(models_metrics.items()), columns=['Model', metric_name])
    
    # 排序
    if sort_values:
        df = df.sort_values(by=metric_name, ascending=False)
    
    # 绘制条形图
    plt.figure(figsize=figsize)
    
    # 使用Seaborn改善视觉效果
    ax = sns.barplot(x='Model', y=metric_name, data=df, palette='viridis')
    
    # 在每个条形上方显示具体数值
    for i, value in enumerate(df[metric_name]):
        ax.text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    if title:
        plt.title(title)
    else:
        plt.title(f'Model Comparison - {metric_name}')
    
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, min(1.0, max(df[metric_name]) * 1.15))  # 设置y轴上限
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.png", bbox_inches='tight')
    plt.savefig(f"{output_dir}/{filename}.pdf", bbox_inches='tight')
    plt.close()


def plot_class_performance(classification_report_df: pd.DataFrame,
                         metric: str = 'f1-score',
                         output_dir: str = 'paper_figures',
                         filename: str = 'class_performance',
                         figsize: Tuple[int, int] = (12, 8)):
    """
    绘制每个类别的性能指标
    
    Args:
        classification_report_df: 分类报告DataFrame
        metric: 要绘制的指标 ('precision', 'recall', 或 'f1-score')
        output_dir: 输出目录
        filename: 输出文件名
        figsize: 图像尺寸
    """
    create_output_dir(output_dir)
    
    # 筛选数据
    df = classification_report_df.iloc[:-3].copy()  # 排除平均行
    df = df.sort_values(by=metric, ascending=False)
    
    # 绘制条形图
    plt.figure(figsize=figsize)
    
    ax = sns.barplot(x=df.index, y=df[metric], palette='coolwarm')
    
    # 在每个条形上方显示具体数值
    for i, value in enumerate(df[metric]):
        ax.text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.title(f'Per-Class {metric.replace("_", " ").title()}')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.xlabel('Class')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}_{metric}.png", bbox_inches='tight')
    plt.savefig(f"{output_dir}/{filename}_{metric}.pdf", bbox_inches='tight')
    plt.close()
    
    # 同时绘制所有指标的对比图
    metrics = ['precision', 'recall', 'f1-score']
    df_plot = df[metrics].copy()
    
    plt.figure(figsize=figsize)
    df_plot.plot(kind='bar', figsize=figsize)
    
    plt.title('Per-Class Performance Metrics')
    plt.ylabel('Score')
    plt.xlabel('Class')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}_all_metrics.png", bbox_inches='tight')
    plt.savefig(f"{output_dir}/{filename}_all_metrics.pdf", bbox_inches='tight')
    plt.close()


def visualize_misclassified_samples(model, dataloader, class_names, 
                                   output_dir='paper_figures',
                                   filename='misclassified_samples',
                                   num_samples=10,
                                   figsize=(15, 10),
                                   device=None):
    """
    可视化错误分类的样本
    
    Args:
        model: PyTorch模型
        dataloader: 数据加载器
        class_names: 类别名称列表
        output_dir: 输出目录
        filename: 输出文件名
        num_samples: 要显示的样本数量
        figsize: 图像尺寸
        device: 运行设备
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    create_output_dir(output_dir)
    model = model.to(device)
    model.eval()
    
    misclassified_images = []
    misclassified_preds = []
    misclassified_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # 查找错误分类的样本
            incorrect_idx = (preds != labels).nonzero(as_tuple=True)[0]
            
            for idx in incorrect_idx:
                if len(misclassified_images) < num_samples:
                    misclassified_images.append(images[idx].cpu())
                    misclassified_preds.append(preds[idx].item())
                    misclassified_labels.append(labels[idx].item())
                else:
                    break
            
            if len(misclassified_images) >= num_samples:
                break
    
    # 计算要绘制的行数和列数
    n_samples = len(misclassified_images)
    if n_samples == 0:
        print("没有找到错误分类的样本！")
        return
    
    n_cols = min(5, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    plt.figure(figsize=figsize)
    
    for i in range(n_samples):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # 将图像从tensor转换为numpy数组并反归一化（如果需要）
        img = misclassified_images[i].permute(1, 2, 0).numpy()
        
        # 尝试反归一化，如果出错则直接使用原始图像
        try:
            # 这里假设使用的是ImageNet标准化参数，根据需要调整
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
        except:
            pass
        
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        
        true_label = class_names[misclassified_labels[i]]
        pred_label = class_names[misclassified_preds[i]]
        
        plt.title(f'True: {true_label}\nPred: {pred_label}', color='red', fontsize=10)
        plt.axis('off')
    
    plt.suptitle('Misclassified Samples', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # 为标题腾出空间
    
    plt.savefig(f"{output_dir}/{filename}.png", bbox_inches='tight')
    plt.savefig(f"{output_dir}/{filename}.pdf", bbox_inches='tight')
    plt.close()


def plot_grad_cam(model, dataloader, class_names, target_layer_name,
                output_dir='paper_figures',
                filename='grad_cam',
                num_samples=5,
                figsize=(15, 8),
                device=None):
    """
    绘制Grad-CAM热力图，显示模型关注的区域
    
    Args:
        model: PyTorch模型
        dataloader: 数据加载器
        class_names: 类别名称列表
        target_layer_name: 目标层名称（通常是最后一个卷积层）
        output_dir: 输出目录
        filename: 输出文件名
        num_samples: 要显示的样本数量
        figsize: 图像尺寸
        device: 运行设备
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("请安装pytorch-grad-cam包：pip install pytorch-grad-cam")
        return
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    create_output_dir(output_dir)
    model = model.to(device)
    model.eval()
    
    # 查找目标层
    target_layer = None
    for name, module in model.named_modules():
        if name == target_layer_name:
            target_layer = module
            break
    
    if target_layer is None:
        print(f"找不到层: {target_layer_name}")
        return
    
    # 初始化GradCAM
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=device.type=='cuda')
    
    # 从数据加载器中获取样本
    images, labels = next(iter(dataloader))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples].to(device)
    
    # 计算输出和预测
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    # 创建图像
    plt.figure(figsize=figsize)
    
    for i in range(min(num_samples, len(images))):
        # 原始图像
        plt.subplot(2, num_samples, i + 1)
        img = images[i].cpu().permute(1, 2, 0).numpy()
        
        # 反归一化
        try:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
        except:
            pass
        
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        true_label = class_names[labels[i].item()]
        pred_label = class_names[preds[i].item()]
        is_correct = labels[i].item() == preds[i].item()
        color = 'green' if is_correct else 'red'
        plt.title(f'True: {true_label}\nPred: {pred_label}', color=color, fontsize=10)
        plt.axis('off')
        
        # Grad-CAM热力图
        plt.subplot(2, num_samples, i + 1 + num_samples)
        
        # 计算目标类别的CAM
        target_category = preds[i].item()  # 使用预测的类别
        grayscale_cam = cam(input_tensor=images[i].unsqueeze(0), target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]  # 移除批次维度
        
        # 叠加热力图
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        plt.imshow(cam_image)
        plt.title('Grad-CAM Heatmap', fontsize=10)
        plt.axis('off')
    
    plt.suptitle('Model Attention Visualization with Grad-CAM', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # 为标题腾出空间
    
    plt.savefig(f"{output_dir}/{filename}.png", bbox_inches='tight')
    plt.savefig(f"{output_dir}/{filename}.pdf", bbox_inches='tight')
    plt.close()


# 使用示例：生成所有评估图表的脚本
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='生成论文级图表')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--data_dir', type=str, required=True, help='验证数据目录')
    parser.add_argument('--output_dir', type=str, default='paper_figures', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--history_path', type=str, help='训练历史CSV文件路径（可选）')
    parser.add_argument('--feature_layer', type=str, help='特征提取层名称（可选）')
    parser.add_argument('--target_layer', type=str, help='Grad-CAM目标层名称（可选）')
    
    args = parser.parse_args()
    
    import torch
    import torch.nn as nn
    from torchvision import models, transforms, datasets
    from torch.utils.data import DataLoader
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载验证数据集
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 获取类别名称
    class_names = val_dataset.classes
    
    # 加载模型
    model = models.resnet18(pretrained=False)
    num_classes = len(class_names)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # 加载模型权重
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # 加载训练历史（如果有）
    history_df = None
    if args.history_path:
        history_df = pd.read_csv(args.history_path)
    
    # 生成所有评估图表
    metrics = generate_evaluation_plots(
        model=model,
        val_loader=val_loader,
        class_names=class_names,
        history_df=history_df,
        feature_layer_name=args.feature_layer,
        output_dir=args.output_dir,
        device=device
    )
    
    # 可视化错误分类的样本
    visualize_misclassified_samples(
        model=model,
        dataloader=val_loader,
        class_names=class_names,
        output_dir=args.output_dir,
        device=device
    )
    
    # 如果指定了目标层，绘制Grad-CAM热力图
    if args.target_layer:
        plot_grad_cam(
            model=model,
            dataloader=val_loader,
            class_names=class_names,
            target_layer_name=args.target_layer,
            output_dir=args.output_dir,
            device=device
        )
    
    # 绘制每个类别的性能指标
    if 'classification_report' in metrics:
        plot_class_performance(
            metrics['classification_report'],
            output_dir=args.output_dir
        )
    
    print(f"全部图表已生成并保存到目录: {args.output_dir}")
    print(f"模型准确率: {metrics['accuracy']:.4f}")
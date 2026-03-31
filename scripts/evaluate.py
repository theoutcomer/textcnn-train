"""
模型评估脚本
支持多维度评估指标和可视化
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, auc,
    hamming_loss, jaccard_score, f1_score
)
from torch.utils.data import DataLoader
import torch

from src.predictor import Predictor
from src.data.dataset import TextDataset, LabelManager, Vocabulary
from src.utils.config import Config


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    计算多标签分类的各项指标
    
    Args:
        y_true: 真实标签 (n_samples, n_classes)
        y_pred: 预测标签 (n_samples, n_classes)
        y_prob: 预测概率 (n_samples, n_classes)
    
    Returns:
        指标字典
    """
    metrics = {}
    
    # Micro 平均（考虑所有标签的总体表现）
    metrics['micro_precision'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['micro_recall'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    # Macro 平均（每个标签平等）
    metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # 其他指标
    metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
    metrics['jaccard_score'] = jaccard_score(y_true, y_pred, average='micro', zero_division=0)
    
    # 每个标签的 F1
    per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics['per_label_f1'] = per_label_f1.tolist()
    
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, label_names: list, save_path: str):
    """
    绘制每个标签的混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        label_names: 标签名称
        save_path: 保存路径
    """
    n_classes = len(label_names)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(min(n_classes, 8)):  # 最多显示8个标签
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{label_names[i]}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, label_names: list, save_path: str):
    """
    绘制 Precision-Recall 曲线
    
    Args:
        y_true: 真实标签
        y_prob: 预测概率
        label_names: 标签名称
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))
    
    n_classes = len(label_names)
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
        avg_precision = auc(recall, precision)
        plt.plot(recall, precision, label=f'{label_names[i]} (AP={avg_precision:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"PR curve saved to: {save_path}")


def plot_label_distribution(y_true: np.ndarray, y_pred: np.ndarray, label_names: list, save_path: str):
    """
    绘制标签分布对比图
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        label_names: 标签名称
        save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 真实分布
    true_counts = y_true.sum(axis=0)
    ax1.bar(range(len(label_names)), true_counts)
    ax1.set_xticks(range(len(label_names)))
    ax1.set_xticklabels(label_names, rotation=45, ha='right')
    ax1.set_title('True Label Distribution')
    ax1.set_ylabel('Count')
    
    # 预测分布
    pred_counts = y_pred.sum(axis=0)
    ax2.bar(range(len(label_names)), pred_counts)
    ax2.set_xticks(range(len(label_names)))
    ax2.set_xticklabels(label_names, rotation=45, ha='right')
    ax2.set_title('Predicted Label Distribution')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Label distribution saved to: {save_path}")


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, label_names: list) -> dict:
    """
    寻找每个标签的最优阈值
    
    Args:
        y_true: 真实标签
        y_prob: 预测概率
        label_names: 标签名称
    
    Returns:
        最优阈值字典
    """
    optimal_thresholds = {}
    
    for i, label in enumerate(label_names):
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in np.arange(0.1, 0.9, 0.05):
            y_pred = (y_prob[:, i] >= threshold).astype(int)
            f1 = f1_score(y_true[:, i], y_pred, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimal_thresholds[label] = {
            'threshold': best_threshold,
            'f1_score': best_f1
        }
    
    return optimal_thresholds


def main():
    parser = argparse.ArgumentParser(description='Evaluate TextCNN model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, required=True,
                        help='Path to vocabulary file')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to labels file')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载配置
    config = Config.from_yaml(args.config)
    
    print("=" * 60)
    print("TextCNN Model Evaluation")
    print("=" * 60)
    
    # 加载预测器
    print("\nLoading model...")
    predictor = Predictor.from_checkpoint(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        label_path=args.labels,
        config=config
    )
    
    model_info = predictor.get_model_info()
    label_names = model_info['labels']
    print(f"Model loaded: {len(label_names)} labels")
    print(f"Labels: {label_names}")
    
    # 加载测试数据
    print(f"\nLoading test data from: {args.test_data}")
    
    if args.test_data.endswith('.json'):
        with open(args.test_data, 'r', encoding='utf-8') as f:
            data = json.load(f)
        texts = [item['text'] for item in data]
        true_labels = [item['labels'] for item in data]
    elif args.test_data.endswith('.csv'):
        df = pd.read_csv(args.test_data)
        texts = df['text'].tolist()
        true_labels = df['labels'].apply(lambda x: x.split(',') if isinstance(x, str) else []).tolist()
    else:
        raise ValueError(f"Unsupported file format: {args.test_data}")
    
    print(f"Test samples: {len(texts)}")
    
    # 批量预测
    print("\nRunning predictions...")
    results = predictor.predict_batch(texts, batch_size=args.batch_size, return_probs=True)
    
    # 准备评估数据
    y_true = np.array([predictor.label_manager.encode(labels) for labels in true_labels])
    y_prob = np.array([list(r['probabilities'].values()) for r in results])
    y_pred = (y_prob >= config.labels.threshold).astype(int)
    
    # 计算指标
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    print(f"\nOverall Metrics:")
    print(f"  Micro F1:    {metrics['micro_f1']:.4f}")
    print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"  Micro Precision: {metrics['micro_precision']:.4f}")
    print(f"  Micro Recall:    {metrics['micro_recall']:.4f}")
    print(f"  Hamming Loss:    {metrics['hamming_loss']:.4f}")
    print(f"  Jaccard Score:   {metrics['jaccard_score']:.4f}")
    
    # 每个标签的 F1
    print(f"\nPer-Label F1 Scores:")
    for i, label in enumerate(label_names):
        print(f"  {label}: {metrics['per_label_f1'][i]:.4f}")
    
    # 寻找最优阈值
    print("\nFinding optimal thresholds...")
    optimal_thresholds = find_optimal_threshold(y_true, y_prob, label_names)
    print("\nOptimal Thresholds:")
    for label, info in optimal_thresholds.items():
        print(f"  {label}: threshold={info['threshold']:.2f}, F1={info['f1_score']:.4f}")
    
    # 保存结果
    results_dict = {
        'overall_metrics': {
            'micro_f1': metrics['micro_f1'],
            'macro_f1': metrics['macro_f1'],
            'micro_precision': metrics['micro_precision'],
            'micro_recall': metrics['micro_recall'],
            'hamming_loss': metrics['hamming_loss'],
            'jaccard_score': metrics['jaccard_score']
        },
        'per_label_f1': {label: metrics['per_label_f1'][i] for i, label in enumerate(label_names)},
        'optimal_thresholds': optimal_thresholds
    }
    
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # 可视化
    if args.visualize:
        print("\nGenerating visualizations...")
        
        # 混淆矩阵
        plot_confusion_matrix(
            y_true, y_pred, label_names,
            os.path.join(args.output_dir, 'confusion_matrix.png')
        )
        
        # PR 曲线
        plot_precision_recall_curve(
            y_true, y_prob, label_names,
            os.path.join(args.output_dir, 'pr_curve.png')
        )
        
        # 标签分布
        plot_label_distribution(
            y_true, y_pred, label_names,
            os.path.join(args.output_dir, 'label_distribution.png')
        )
        
        print(f"Visualizations saved to: {args.output_dir}")
    
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)


if __name__ == '__main__':
    from sklearn.metrics import precision_score, recall_score
    main()

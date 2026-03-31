"""
训练脚本
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import pandas as pd
from torch.utils.data import DataLoader

from src.models.textcnn import TextCNN
from src.data.dataset import Vocabulary, LabelManager, TextDataset
from src.trainer import Trainer
from src.utils.config import Config


def load_data(data_path: str, label_manager: LabelManager):
    """
    加载数据
    支持 CSV 和 JSON 格式
    
    CSV 格式: text,labels (labels 为逗号分隔的标签)
    JSON 格式: [{"text": "...", "labels": ["标签1", "标签2"]}]
    """
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        texts = df['text'].tolist()
        label_lists = df['labels'].apply(lambda x: x.split(',') if isinstance(x, str) else []).tolist()
    elif data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        texts = [item['text'] for item in data]
        label_lists = [item['labels'] for item in data]
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    # 编码标签
    labels = [label_manager.encode(lbls) for lbls in label_lists]
    
    return texts, labels


def main():
    parser = argparse.ArgumentParser(description='Train TextCNN model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training data')
    parser.add_argument('--val_data', type=str, default=None,
                        help='Path to validation data')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save model checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    # 加载配置
    config = Config.from_yaml(args.config)
    
    # 覆盖配置
    if args.save_dir:
        config.training.save_dir = args.save_dir
    config.data.train_data_path = args.train_data
    config.data.val_data_path = args.val_data
    
    print("=" * 50)
    print("TextCNN Training")
    print("=" * 50)
    print(f"Config: {args.config}")
    print(f"Train data: {args.train_data}")
    print(f"Val data: {args.val_data}")
    print(f"Save dir: {config.training.save_dir}")
    
    # 初始化标签管理器
    label_manager = LabelManager()
    label_manager.add_labels(config.labels.labels)
    print(f"\nLabels ({len(label_manager)}): {label_manager.get_labels()}")
    
    # 加载训练数据
    print("\nLoading training data...")
    train_texts, train_labels = load_data(args.train_data, label_manager)
    print(f"Training samples: {len(train_texts)}")
    
    # 构建词表
    print("\nBuilding vocabulary...")
    vocab = Vocabulary(min_freq=config.data.min_freq, max_size=config.model.vocab_size)
    vocab.build_vocab(train_texts)
    print(f"Vocabulary size: {len(vocab)}")
    
    # 保存词表和标签
    os.makedirs(config.training.save_dir, exist_ok=True)
    vocab.save(config.data.vocab_path)
    label_manager.save(config.data.label_path)
    print(f"Vocabulary saved to: {config.data.vocab_path}")
    print(f"Labels saved to: {config.data.label_path}")
    
    # 创建数据集
    train_dataset = TextDataset(
        texts=train_texts,
        labels=train_labels,
        vocab=vocab,
        max_len=config.model.max_len
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # 验证数据
    val_loader = None
    if args.val_data:
        print("\nLoading validation data...")
        val_texts, val_labels = load_data(args.val_data, label_manager)
        print(f"Validation samples: {len(val_texts)}")
        
        val_dataset = TextDataset(
            texts=val_texts,
            labels=val_labels,
            vocab=vocab,
            max_len=config.model.max_len
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=0
        )
    
    # 创建模型
    print("\nCreating model...")
    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=config.model.embed_dim,
        num_classes=len(label_manager),
        filter_sizes=config.model.filter_sizes,
        num_filters=config.model.num_filters,
        dropout=config.model.dropout
    )
    
    model_info = model.get_model_info()
    print(f"Model parameters: {model_info['total_params']:,}")
    print(f"Trainable parameters: {model_info['trainable_params']:,}")
    
    # 创建训练器
    trainer = Trainer(model, config.training)
    
    # 断点续训
    start_epoch = 0
    if args.resume:
        print(f"\nResuming training from: {args.resume}")
        checkpoint = trainer.load_checkpoint(args.resume)
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming from epoch {start_epoch + 1}")
    
    # 开始训练
    print("\n" + "=" * 50)
    print("Starting training...")
    if args.resume:
        print(f"(Resumed from epoch {start_epoch + 1})")
    print("=" * 50)
    
    history = trainer.train(train_loader, val_loader, start_epoch=start_epoch)
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Best validation F1: {trainer.best_val_f1:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    main()

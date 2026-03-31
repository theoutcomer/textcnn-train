"""云端 GPU 训练脚本 - 支持混合精度、分布式训练"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import time
from datetime import datetime

from src.models.textcnn import TextCNN
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.training.trainer import Trainer
from src.data.dataset import TextDataset
from src.utils.vocab import Vocabulary
from src.utils.label_mapper import LabelMapper
from src.utils.logger import setup_logger
from src.utils.metrics import MetricsTracker


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_gpu():
    """设置 GPU 环境"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")
    
    device = torch.device("cuda")
    gpu_count = torch.cuda.device_count()
    print(f"✓ GPU 可用: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU 数量: {gpu_count}")
    print(f"✓ CUDA 版本: {torch.version.cuda}")
    
    # 设置 cudnn 加速
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    return device, gpu_count


def prepare_data(config: dict, label_mapper: LabelMapper):
    """准备数据加载器"""
    data_config = config['data']
    gpu_config = config['gpu']
    
    # 加载数据集
    train_dataset = TextDataset(
        data_path=data_config['train_path'],
        vocab=None,  # 会在训练时构建
        label_mapper=label_mapper,
        max_len=config['training']['max_len'],
        augment=config['augmentation']['enabled']
    )
    
    # 划分训练/验证/测试集
    total_size = len(train_dataset)
    val_size = int(total_size * data_config['val_split'])
    test_size = int(total_size * data_config['test_split'])
    train_size = total_size - val_size - test_size
    
    train_set, val_set, test_set = torch.utils.data.random_split(
        train_dataset, [train_size, val_size, test_size]
    )
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_set,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=gpu_config['num_workers'],
        pin_memory=gpu_config['pin_memory']
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=gpu_config['num_workers'],
        pin_memory=gpu_config['pin_memory']
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=gpu_config['num_workers'],
        pin_memory=gpu_config['pin_memory']
    )
    
    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, scaler, device, config):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    metrics_tracker = MetricsTracker()
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # 混合精度训练
        if config['gpu']['mixed_precision']:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            if config['training'].get('gradient_clip'):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config['training']['gradient_clip']
                )
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            if config['training'].get('gradient_clip'):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training']['gradient_clip']
                )
            
            optimizer.step()
        
        total_loss += loss.item()
        
        # 计算指标
        probs = torch.sigmoid(outputs)
        predictions = (probs > 0.5).float()
        metrics_tracker.update(predictions, labels)
        
        # 打印进度
        if batch_idx % config['logging']['log_every'] == 0:
            print(f"  Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
    
    metrics = metrics_tracker.compute()
    metrics['loss'] = total_loss / len(train_loader)
    
    return metrics


def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    metrics_tracker = MetricsTracker()
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()
            metrics_tracker.update(predictions, labels)
    
    metrics = metrics_tracker.compute()
    metrics['loss'] = total_loss / len(val_loader)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Cloud GPU Training')
    parser.add_argument('--config', type=str, default='configs/cloud_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--data_path', type=str, default=None,
                        help='训练数据路径（覆盖配置）')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    if args.data_path:
        config['data']['train_path'] = args.data_path
    
    print("="*60)
    print("云端 GPU 训练启动")
    print("="*60)
    print(f"实验名称: {config['experiment']['name']}")
    print(f"配置: {args.config}")
    print(f"数据: {config['data']['train_path']}")
    
    # 设置 GPU
    device, gpu_count = setup_gpu()
    
    # 创建输出目录
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    
    # 准备标签映射
    label_mapper = LabelMapper()
    # TODO: 从数据加载标签
    
    # 准备数据
    print("\n准备数据...")
    train_loader, val_loader, test_loader = prepare_data(config, label_mapper)
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")
    
    # 创建模型
    print("\n创建模型...")
    model_config = config['model']
    model = TextCNN(
        vocab_size=model_config['vocab_size'],
        embed_dim=model_config['embed_dim'],
        num_classes=model_config['num_classes'],
        filter_sizes=model_config['filter_sizes'],
        num_filters=model_config['num_filters'],
        dropout=model_config['dropout']
    )
    model = model.to(device)
    
    # 多 GPU 支持
    if gpu_count > 1:
        model = nn.DataParallel(model)
        print(f"使用 {gpu_count} 个 GPU 并行训练")
    
    # 优化器和学习率调度
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    if config['training']['scheduler']['type'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['scheduler']['T_max']
        )
    else:
        scheduler = None
    
    # 损失函数
    criterion = nn.BCEWithLogitsLoss()
    
    # 混合精度训练
    scaler = GradScaler() if config['gpu']['mixed_precision'] else None
    
    # 训练循环
    print("\n开始训练...")
    best_val_f1 = 0
    patience_counter = 0
    
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        print("-"*40)
        
        # 训练
        start_time = time.time()
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, 
            scaler, device, config
        )
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device)
        
        # 学习率调度
        if scheduler:
            scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # 打印结果
        print(f"  Train Loss: {train_metrics['loss']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # 保存最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if gpu_count > 1 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1,
                'config': config
            }
            
            save_path = os.path.join(
                config['logging']['checkpoint_dir'], 
                'best_model.pt'
            )
            torch.save(checkpoint, save_path)
            print(f"  ✓ 保存最佳模型 (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
        
        # 定期保存
        if (epoch + 1) % config['logging']['save_every'] == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if gpu_count > 1 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_metrics['f1'],
                'config': config
            }
            save_path = os.path.join(
                config['logging']['checkpoint_dir'],
                f'checkpoint_epoch_{epoch+1}.pt'
            )
            torch.save(checkpoint, save_path)
        
        # Early Stopping
        if patience_counter >= config['training']['early_stopping']['patience']:
            print(f"\n早停触发，训练结束")
            break
    
    print("\n" + "="*60)
    print("训练完成!")
    print(f"最佳验证 F1: {best_val_f1:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()

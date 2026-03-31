"""
训练引擎
支持模型训练、验证、早停、检查点保存
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os
import json
from typing import Dict, Optional
from tqdm import tqdm
import numpy as np

from .models.textcnn import TextCNN
from .utils.config import TrainingConfig


class Trainer:
    """
    模型训练器
    """
    
    def __init__(
        self,
        model: TextCNN,
        config: TrainingConfig,
        device: str = None
    ):
        """
        初始化训练器
        
        Args:
            model: TextCNN 模型
            config: 训练配置
            device: 训练设备
        """
        self.model = model
        self.config = config
        
        # 设置设备
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # 损失函数 - 多标签分类使用 BCEWithLogitsLoss
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=2
        )
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # 训练状态
        self.best_val_f1 = 0.0
        self.early_stopping_counter = 0
        self.global_step = 0
        
        # 创建保存目录
        os.makedirs(config.save_dir, exist_ok=True)
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        训练一个 epoch
        
        Args:
            train_loader: 训练数据加载器
        
        Returns:
            训练指标字典
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(input_ids)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 记录
            total_loss += loss.item()
            preds = torch.sigmoid(outputs).cpu().detach().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            self.global_step += 1
            
            # 更新进度条
            pbar.set_postfix({'loss': loss.item()})
        
        # 计算指标
        all_preds = np.array(all_preds) > 0.5
        all_labels = np.array(all_labels)
        
        metrics = self._calculate_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(train_loader)
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
        
        Returns:
            验证指标字典
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        all_preds = np.array(all_preds) > 0.5
        all_labels = np.array(all_labels)
        
        metrics = self._calculate_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(val_loader)
        
        return metrics
    
    def _calculate_metrics(self, labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            labels: 真实标签
            preds: 预测标签
        
        Returns:
            指标字典
        """
        # 多标签分类指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='micro', zero_division=0
        )
        
        # 每个标签的准确率（所有标签都预测正确才算正确）
        exact_match = accuracy_score(labels, preds)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'exact_match': exact_match
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        start_epoch: int = 0
    ) -> Dict[str, list]:
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            start_epoch: 起始轮次（用于断点续训）
        
        Returns:
            训练历史记录
        """
        history = {
            'train_loss': [],
            'train_f1': [],
            'val_loss': [],
            'val_f1': []
        }
        
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(start_epoch, self.config.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            
            # 训练
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])
            history['train_f1'].append(train_metrics['f1'])
            
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"F1: {train_metrics['f1']:.4f}, "
                  f"Precision: {train_metrics['precision']:.4f}, "
                  f"Recall: {train_metrics['recall']:.4f}")
            
            # 记录到 TensorBoard
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('F1/train', train_metrics['f1'], epoch)
            
            # 验证
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                history['val_loss'].append(val_metrics['loss'])
                history['val_f1'].append(val_metrics['f1'])
                
                print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                      f"F1: {val_metrics['f1']:.4f}, "
                      f"Precision: {val_metrics['precision']:.4f}, "
                      f"Recall: {val_metrics['recall']:.4f}")
                
                # 记录到 TensorBoard
                self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
                self.writer.add_scalar('F1/val', val_metrics['f1'], epoch)
                
                # 学习率调度
                self.scheduler.step(val_metrics['f1'])
                
                # 保存最佳模型
                if val_metrics['f1'] > self.best_val_f1:
                    self.best_val_f1 = val_metrics['f1']
                    self.save_checkpoint('best_model.pt', epoch=epoch)
                    self.early_stopping_counter = 0
                    print(f"✓ New best model saved! F1: {self.best_val_f1:.4f}")
                else:
                    self.early_stopping_counter += 1
                
                # 早停检查
                if self.early_stopping_counter >= self.config.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
            
            # 保存最新模型
            self.save_checkpoint('latest_model.pt', epoch=epoch)
        
        self.writer.close()
        return history
    
    def save_checkpoint(self, filename: str, epoch: int = None):
        """
        保存模型检查点
        
        Args:
            filename: 文件名
            epoch: 当前轮次
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_f1': self.best_val_f1,
            'global_step': self.global_step,
            'model_config': self.model.get_model_info(),
            'epoch': epoch
        }
        
        path = os.path.join(self.config.save_dir, filename)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, filename: str):
        """
        加载模型检查点
        
        Args:
            filename: 文件名
        
        Returns:
            checkpoint 字典
        """
        path = filename if os.path.exists(filename) else os.path.join(self.config.save_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_f1 = checkpoint['best_val_f1']
        self.global_step = checkpoint['global_step']
        
        return checkpoint

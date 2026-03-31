"""
增量学习模块
支持在线更新和新增标签
"""
import torch
import torch.nn as nn
from typing import List, Dict, Optional
import copy


class IncrementalLearner:
    """
    增量学习器
    支持新增标签和在线更新
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        初始化增量学习器
        
        Args:
            model: 基础模型
            device: 训练设备
        """
        self.base_model = model
        self.device = torch.device(device)
        self.base_model.to(self.device)
        
        # 记录已学习的标签
        self.learned_labels = set()
        self.label_to_idx = {}
    
    def add_new_labels(
        self,
        new_labels: List[str],
        freeze_base: bool = True
    ) -> nn.Module:
        """
        添加新标签
        扩展模型输出层以支持新标签
        
        Args:
            new_labels: 新标签列表
            freeze_base: 是否冻结基础模型参数
        
        Returns:
            更新后的模型
        """
        # 记录新标签
        for label in new_labels:
            if label not in self.learned_labels:
                idx = len(self.learned_labels)
                self.learned_labels.add(label)
                self.label_to_idx[label] = idx
        
        # 更新模型输出层
        old_num_classes = self.base_model.num_classes
        new_num_classes = len(self.learned_labels)
        
        if new_num_classes > old_num_classes:
            self.base_model.update_num_classes(new_num_classes)
        
        # 冻结基础模型（可选）
        if freeze_base:
            for name, param in self.base_model.named_parameters():
                if 'fc' not in name:  # 只冻结非分类层
                    param.requires_grad = False
        
        return self.base_model
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """
        获取可训练参数
        用于增量学习时只优化特定参数
        
        Returns:
            可训练参数列表
        """
        return [p for p in self.base_model.parameters() if p.requires_grad]
    
    def elastic_weight_consolidation(
        self,
        dataloader,
        important_weight: float = 1000.0
    ) -> Dict[str, torch.Tensor]:
        """
        弹性权重巩固 (EWC)
        计算参数的重要性，用于防止灾难性遗忘
        
        Args:
            dataloader: 旧任务数据加载器
            important_weight: 重要性权重系数
        
        Returns:
            参数重要性字典
        """
        self.base_model.eval()
        
        # 计算 Fisher 信息矩阵对角线
        fisher_dict = {}
        for name, param in self.base_model.named_parameters():
            fisher_dict[name] = torch.zeros_like(param)
        
        # 在旧数据上计算梯度
        for batch in dataloader:
            self.base_model.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.base_model(input_ids)
            loss = nn.BCEWithLogitsLoss()(outputs, labels)
            loss.backward()
            
            # 累积梯度平方（Fisher 信息）
            for name, param in self.base_model.named_parameters():
                if param.grad is not None:
                    fisher_dict[name] += param.grad.pow(2)
        
        # 平均
        num_batches = len(dataloader)
        for name in fisher_dict:
            fisher_dict[name] /= num_batches
            fisher_dict[name] *= important_weight
        
        return fisher_dict
    
    def train_with_ewc(
        self,
        new_dataloader,
        old_fisher_dict: Dict[str, torch.Tensor],
        old_params: Dict[str, torch.Tensor],
        optimizer,
        epochs: int = 5
    ):
        """
        使用 EWC 进行增量训练
        防止在旧任务上的性能下降
        
        Args:
            new_dataloader: 新任务数据加载器
            old_fisher_dict: 旧任务的 Fisher 信息
            old_params: 旧任务的最优参数
            optimizer: 优化器
            epochs: 训练轮数
        """
        self.base_model.train()
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for batch in new_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                # 新任务损失
                outputs = self.base_model(input_ids)
                new_loss = criterion(outputs, labels)
                
                # EWC 正则化损失
                ewc_loss = 0.0
                for name, param in self.base_model.named_parameters():
                    if name in old_fisher_dict:
                        _loss = old_fisher_dict[name] * (param - old_params[name]).pow(2)
                        ewc_loss += _loss.sum()
                
                # 总损失
                total_loss_batch = new_loss + ewc_loss
                total_loss_batch.backward()
                optimizer.step()
                
                total_loss += total_loss_batch.item()
            
            avg_loss = total_loss / len(new_dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def knowledge_distillation(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        dataloader,
        temperature: float = 2.0,
        alpha: float = 0.5,
        epochs: int = 5,
        lr: float = 0.001
    ):
        """
        知识蒸馏
        使用旧模型作为教师模型，训练新模型
        
        Args:
            teacher_model: 教师模型（旧模型）
            student_model: 学生模型（新模型）
            dataloader: 训练数据
            temperature: 蒸馏温度
            alpha: 蒸馏损失与硬标签损失的权重
            epochs: 训练轮数
            lr: 学习率
        """
        teacher_model.eval()
        student_model.train()
        
        optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                # 教师模型输出
                with torch.no_grad():
                    teacher_logits = teacher_model(input_ids)
                    teacher_probs = torch.sigmoid(teacher_logits / temperature)
                
                # 学生模型输出
                student_logits = student_model(input_ids)
                student_probs = torch.sigmoid(student_logits / temperature)
                
                # 蒸馏损失（KL 散度）
                distill_loss = nn.KLDivLoss(reduction='batchmean')(
                    torch.log(student_probs + 1e-10),
                    teacher_probs
                ) * (temperature ** 2)
                
                # 硬标签损失
                hard_loss = nn.BCEWithLogitsLoss()(student_logits, labels)
                
                # 总损失
                loss = alpha * distill_loss + (1 - alpha) * hard_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def save_checkpoint(self, path: str, metadata: Dict = None):
        """
        保存增量学习检查点
        
        Args:
            path: 保存路径
            metadata: 额外元数据
        """
        checkpoint = {
            'model_state_dict': self.base_model.state_dict(),
            'learned_labels': list(self.learned_labels),
            'label_to_idx': self.label_to_idx,
            'metadata': metadata or {}
        }
        
        torch.save(checkpoint, path)
        print(f"Incremental learning checkpoint saved to: {path}")
    
    def load_checkpoint(self, path: str):
        """
        加载增量学习检查点
        
        Args:
            path: 检查点路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.base_model.load_state_dict(checkpoint['model_state_dict'])
        self.learned_labels = set(checkpoint['learned_labels'])
        self.label_to_idx = checkpoint['label_to_idx']
        
        print(f"Incremental learning checkpoint loaded from: {path}")
        print(f"Learned labels: {self.learned_labels}")

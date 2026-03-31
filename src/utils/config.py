"""
配置管理
使用 Pydantic 进行配置验证
"""
from pydantic import BaseModel, Field
from typing import List, Optional
import yaml
import os


class ModelConfig(BaseModel):
    """模型配置"""
    vocab_size: int = Field(default=50000, description="词表大小")
    embed_dim: int = Field(default=300, description="词向量维度")
    filter_sizes: List[int] = Field(default=[2, 3, 4, 5], description="卷积核尺寸")
    num_filters: int = Field(default=100, description="每种卷积核的数量")
    dropout: float = Field(default=0.5, ge=0.0, le=1.0, description="Dropout比率")
    max_len: int = Field(default=512, description="最大序列长度")


class TrainingConfig(BaseModel):
    """训练配置"""
    batch_size: int = Field(default=64, description="批次大小")
    epochs: int = Field(default=20, description="训练轮数")
    learning_rate: float = Field(default=0.001, description="学习率")
    weight_decay: float = Field(default=1e-5, description="权重衰减")
    early_stopping_patience: int = Field(default=5, description="早停耐心值")
    save_dir: str = Field(default="./checkpoints", description="模型保存目录")
    log_dir: str = Field(default="./logs", description="日志目录")
    device: str = Field(default="auto", description="训练设备: auto/cpu/cuda")


class DataConfig(BaseModel):
    """数据配置"""
    train_data_path: Optional[str] = Field(default=None, description="训练数据路径")
    val_data_path: Optional[str] = Field(default=None, description="验证数据路径")
    test_data_path: Optional[str] = Field(default=None, description="测试数据路径")
    vocab_path: str = Field(default="./vocab.pkl", description="词表保存路径")
    label_path: str = Field(default="./labels.pkl", description="标签配置保存路径")
    min_freq: int = Field(default=2, description="最小词频")


class LabelConfig(BaseModel):
    """标签配置 - 支持动态扩展"""
    labels: List[str] = Field(default=[], description="标签列表")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="预测阈值")
    
    def add_label(self, label: str):
        """动态添加标签"""
        if label not in self.labels:
            self.labels.append(label)
    
    def remove_label(self, label: str):
        """移除标签"""
        if label in self.labels:
            self.labels.remove(label)


class Config(BaseModel):
    """总配置"""
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    labels: LabelConfig = Field(default_factory=LabelConfig)
    
    @classmethod
    def from_yaml(cls, path: str):
        """从 YAML 文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str):
        """保存配置到 YAML 文件"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.model_dump(), f, allow_unicode=True, default_flow_style=False)
    
    def update_labels(self, new_labels: List[str]):
        """
        更新标签配置
        用于动态添加新标签
        """
        for label in new_labels:
            self.labels.add_label(label)
        # 更新模型输出的类别数
        self.model.vocab_size = max(self.model.vocab_size, len(self.labels.labels))

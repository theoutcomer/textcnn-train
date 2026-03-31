"""
TextCNN 模型定义
支持动态标签数量调整
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """
    TextCNN 文本分类模型
    
    架构:
    - 嵌入层
    - 多尺度卷积层 (2,3,4,5)
    - Max-over-time pooling
    - Dropout
    - 全连接输出层
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        filter_sizes: list = None,
        num_filters: int = 100,
        dropout: float = 0.5,
        pretrained_embedding: torch.Tensor = None,
        freeze_embedding: bool = False
    ):
        """
        初始化 TextCNN 模型
        
        Args:
            vocab_size: 词表大小
            embed_dim: 词向量维度
            num_classes: 分类标签数量（支持动态调整）
            filter_sizes: 卷积核尺寸列表，默认 [2, 3, 4, 5]
            num_filters: 每种尺寸卷积核的数量
            dropout: Dropout 比率
            pretrained_embedding: 预训练词向量，可选
            freeze_embedding: 是否冻结词向量
        """
        super(TextCNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.filter_sizes = filter_sizes or [2, 3, 4, 5]
        self.num_filters = num_filters
        
        # 嵌入层
        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embedding,
                freeze=freeze_embedding
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 卷积层 - 多尺度卷积
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=fs
            )
            for fs in self.filter_sizes
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 全连接层 - 输出维度由 num_classes 决定，支持动态调整
        self.fc = nn.Linear(len(self.filter_sizes) * num_filters, num_classes)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状 (batch_size, seq_len)
        
        Returns:
            输出张量，形状 (batch_size, num_classes)
        """
        # 嵌入层: (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        x = self.embedding(x)
        
        # 调整维度: (batch_size, seq_len, embed_dim) -> (batch_size, embed_dim, seq_len)
        x = x.permute(0, 2, 1)
        
        # 多尺度卷积 + ReLU + Max-over-time pooling
        conv_results = []
        for conv in self.convs:
            # 卷积: (batch_size, embed_dim, seq_len) -> (batch_size, num_filters, new_seq_len)
            conv_out = F.relu(conv(x))
            # Max pooling: (batch_size, num_filters, new_seq_len) -> (batch_size, num_filters)
            pooled = F.max_pool1d(conv_out, conv_out.shape[2]).squeeze(2)
            conv_results.append(pooled)
        
        # 拼接所有卷积结果: (batch_size, len(filter_sizes) * num_filters)
        x = torch.cat(conv_results, dim=1)
        
        # Dropout
        x = self.dropout(x)
        
        # 全连接层: (batch_size, len(filter_sizes) * num_filters) -> (batch_size, num_classes)
        x = self.fc(x)
        
        return x
    
    def update_num_classes(self, new_num_classes: int):
        """
        动态更新输出类别数量
        用于新增标签时扩展模型输出维度
        
        Args:
            new_num_classes: 新的类别数量
        """
        if new_num_classes <= self.num_classes:
            return
        
        old_fc = self.fc
        old_weight = old_fc.weight.data
        old_bias = old_fc.bias.data
        
        # 创建新的全连接层
        self.fc = nn.Linear(
            len(self.filter_sizes) * self.num_filters,
            new_num_classes
        )
        
        # 复制旧的权重和偏置
        with torch.no_grad():
            self.fc.weight[:self.num_classes] = old_weight
            self.fc.bias[:self.num_classes] = old_bias
        
        self.num_classes = new_num_classes
    
    def get_model_info(self) -> dict:
        """
        获取模型信息
        
        Returns:
            包含模型配置信息的字典
        """
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_classes": self.num_classes,
            "filter_sizes": self.filter_sizes,
            "num_filters": self.num_filters,
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

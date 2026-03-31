"""
BERT + TextCNN 混合模型
使用BERT获取上下文词向量，再用TextCNN进行分类
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class BertTextCNN(nn.Module):
    """
    BERT + TextCNN 模型
    利用BERT的预训练知识，提升分类性能
    """
    
    def __init__(
        self,
        num_classes: int,
        bert_model_name: str = 'bert-base-chinese',
        filter_sizes: list = [2, 3, 4, 5],
        num_filters: int = 128,
        dropout: float = 0.5,
        freeze_bert: bool = True  # 是否冻结BERT参数
    ):
        super(BertTextCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # 加载预训练BERT
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_dim = self.bert.config.hidden_size  # 768
        
        # 是否冻结BERT
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # TextCNN层
        self.convs = nn.ModuleList([
            nn.Conv1d(self.bert_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # Dropout和全连接
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Returns:
            logits: [batch_size, num_classes]
        """
        # BERT编码 [batch_size, seq_len, bert_dim]
        with torch.no_grad() if not self.training else torch.enable_grad():
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            x = bert_outputs.last_hidden_state  # [batch, seq_len, 768]
        
        # 转置为 [batch, channels, seq_len] 以适应Conv1d
        x = x.permute(0, 2, 1)  # [batch, 768, seq_len]
        
        # 多尺度卷积 + Max-over-time pooling
        conv_outputs = []
        for conv in self.convs:
            # conv: [batch, num_filters, seq_len - kernel_size + 1]
            conv_out = F.relu(conv(x))
            # max pooling: [batch, num_filters]
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        
        # 拼接所有卷积结果
        x = torch.cat(conv_outputs, dim=1)  # [batch, len(filter_sizes) * num_filters]
        
        # Dropout和分类
        x = self.dropout(x)
        logits = self.fc(x)  # [batch, num_classes]
        
        return logits
    
    def update_num_classes(self, new_num_classes: int):
        """动态增加类别数"""
        old_fc = self.fc
        old_weight = old_fc.weight.data
        old_bias = old_fc.bias.data if old_fc.bias is not None else None
        
        # 创建新的全连接层
        in_features = old_fc.in_features
        self.fc = nn.Linear(in_features, new_num_classes)
        
        # 复制旧权重
        with torch.no_grad():
            self.fc.weight[:self.num_classes] = old_weight
            if old_bias is not None:
                self.fc.bias[:self.num_classes] = old_bias
        
        self.num_classes = new_num_classes


class BertTokenizerWrapper:
    """BERT分词器包装类"""
    
    def __init__(self, model_name: str = 'bert-base-chinese', max_len: int = 512):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_len = max_len
    
    def encode(self, text: str):
        """编码文本"""
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)
    
    def batch_encode(self, texts: list):
        """批量编码"""
        encodings = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encodings['input_ids'], encodings['attention_mask']

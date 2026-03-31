"""
数据集定义和数据加载
"""
import torch
from torch.utils.data import Dataset
import jieba
import pickle
import os
from typing import List, Dict, Tuple


class Vocabulary:
    """
    词表管理类
    支持从文本构建词表、保存和加载
    """
    
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    
    def __init__(self, min_freq: int = 2, max_size: int = 50000):
        """
        初始化词表
        
        Args:
            min_freq: 最小词频，低于此频率的词将被忽略
            max_size: 词表最大大小
        """
        self.min_freq = min_freq
        self.max_size = max_size
        
        # 特殊token
        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1
        }
        self.idx2word = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
        self.word_counts = {}
        
    def build_vocab(self, texts: List[str]):
        """
        从文本列表构建词表
        
        Args:
            texts: 文本列表
        """
        # 统计词频
        for text in texts:
            words = self.tokenize(text)
            for word in words:
                self.word_counts[word] = self.word_counts.get(word, 0) + 1
        
        # 按词频排序，构建词表
        sorted_words = sorted(
            self.word_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for word, count in sorted_words:
            if count >= self.min_freq and len(self.word2idx) < self.max_size:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def tokenize(self, text: str) -> List[str]:
        """
        中文分词
        
        Args:
            text: 输入文本
        
        Returns:
            分词结果列表
        """
        return list(jieba.cut(text.strip()))
    
    def encode(self, text: str, max_len: int = None) -> List[int]:
        """
        将文本编码为索引序列
        
        Args:
            text: 输入文本
            max_len: 最大序列长度，超出截断，不足填充
        
        Returns:
            索引列表
        """
        words = self.tokenize(text)
        indices = [self.word2idx.get(w, self.word2idx[self.UNK_TOKEN]) for w in words]
        
        if max_len:
            if len(indices) < max_len:
                indices.extend([self.word2idx[self.PAD_TOKEN]] * (max_len - len(indices)))
            else:
                indices = indices[:max_len]
        
        return indices
    
    def __len__(self):
        return len(self.word2idx)
    
    def save(self, path: str):
        """保存词表到文件"""
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_counts': self.word_counts,
                'min_freq': self.min_freq,
                'max_size': self.max_size
            }, f)
    
    @classmethod
    def load(cls, path: str):
        """从文件加载词表"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        vocab = cls(min_freq=data['min_freq'], max_size=data['max_size'])
        vocab.word2idx = data['word2idx']
        vocab.idx2word = data['idx2word']
        vocab.word_counts = data['word_counts']
        return vocab


class TextDataset(Dataset):
    """
    文本分类数据集
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[List[int]],
        vocab: Vocabulary,
        max_len: int = 512
    ):
        """
        初始化数据集
        
        Args:
            texts: 文本列表
            labels: 标签列表（多标签，one-hot 编码）
            vocab: 词表对象
            max_len: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 编码文本
        encoded = self.vocab.encode(text, self.max_len)
        
        return {
            'input_ids': torch.tensor(encoded, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.float)
        }


class LabelManager:
    """
    标签管理类
    支持动态添加标签
    """
    
    def __init__(self):
        self.label2idx = {}
        self.idx2label = {}
    
    def add_labels(self, labels: List[str]):
        """
        添加新标签
        
        Args:
            labels: 标签名称列表
        """
        for label in labels:
            if label not in self.label2idx:
                idx = len(self.label2idx)
                self.label2idx[label] = idx
                self.idx2label[idx] = label
    
    def encode(self, labels: List[str]) -> List[int]:
        """
        将标签列表编码为 one-hot 向量
        
        Args:
            labels: 标签名称列表
        
        Returns:
            one-hot 编码列表
        """
        encoded = [0] * len(self.label2idx)
        for label in labels:
            if label in self.label2idx:
                encoded[self.label2idx[label]] = 1
        return encoded
    
    def decode(self, encoded: List[int], threshold: float = 0.5) -> List[str]:
        """
        将预测结果解码为标签列表
        
        Args:
            encoded: 预测概率或 one-hot 向量
            threshold: 概率阈值
        
        Returns:
            标签名称列表
        """
        labels = []
        for idx, prob in enumerate(encoded):
            if prob >= threshold:
                labels.append(self.idx2label.get(idx, f"label_{idx}"))
        return labels
    
    def __len__(self):
        return len(self.label2idx)
    
    def get_labels(self) -> List[str]:
        """获取所有标签名称"""
        return [self.idx2label[i] for i in range(len(self.label2idx))]
    
    def save(self, path: str):
        """保存标签配置"""
        with open(path, 'wb') as f:
            pickle.dump({
                'label2idx': self.label2idx,
                'idx2label': self.idx2label
            }, f)
    
    @classmethod
    def load(cls, path: str):
        """加载标签配置"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        manager = cls()
        manager.label2idx = data['label2idx']
        manager.idx2label = data['idx2label']
        return manager

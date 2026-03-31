"""
推理服务
支持单条/批量预测、模型加载、标签解码
"""
import torch
import numpy as np
from typing import List, Dict, Union, Optional
import os

from .models.textcnn import TextCNN
from .data.dataset import Vocabulary, LabelManager
from .utils.config import Config


class Predictor:
    """
    文本分类预测器
    """
    
    def __init__(
        self,
        model: TextCNN,
        vocab: Vocabulary,
        label_manager: LabelManager,
        device: str = None,
        threshold: float = 0.5
    ):
        """
        初始化预测器
        
        Args:
            model: TextCNN 模型
            vocab: 词表
            label_manager: 标签管理器
            device: 推理设备
            threshold: 预测阈值
        """
        self.model = model
        self.vocab = vocab
        self.label_manager = label_manager
        self.threshold = threshold
        
        # 设置设备
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict(
        self,
        texts: Union[str, List[str]],
        return_probs: bool = False
    ) -> Union[Dict, List[Dict]]:
        """
        预测文本标签
        
        Args:
            texts: 单条文本或文本列表
            return_probs: 是否返回概率值
        
        Returns:
            预测结果字典或列表
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        # 编码文本
        encoded_texts = [
            self.vocab.encode(text, max_len=self.model.embedding.num_embeddings)
            for text in texts
        ]
        
        # 填充到相同长度
        max_len = max(len(e) for e in encoded_texts)
        padded = []
        for encoded in encoded_texts:
            if len(encoded) < max_len:
                encoded.extend([0] * (max_len - len(encoded)))
            padded.append(encoded)
        
        # 转换为张量
        input_tensor = torch.tensor(padded, dtype=torch.long).to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()
        
        # 解码结果
        results = []
        for prob in probs:
            predicted_labels = self.label_manager.decode(prob, self.threshold)
            result = {
                'labels': predicted_labels,
                'label_indices': [i for i, p in enumerate(prob) if p >= self.threshold]
            }
            
            if return_probs:
                result['probabilities'] = {
                    self.label_manager.idx2label.get(i, f"label_{i}"): float(p)
                    for i, p in enumerate(prob)
                }
            
            results.append(result)
        
        return results[0] if single_input else results
    
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        return_probs: bool = False
    ) -> List[Dict]:
        """
        批量预测（带进度条）
        
        Args:
            texts: 文本列表
            batch_size: 批次大小
            return_probs: 是否返回概率值
        
        Returns:
            预测结果列表
        """
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = self.predict(batch_texts, return_probs=return_probs)
            all_results.extend(batch_results)
        
        return all_results
    
    def update_threshold(self, threshold: float):
        """
        更新预测阈值
        
        Args:
            threshold: 新阈值
        """
        self.threshold = threshold
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'model_config': self.model.get_model_info(),
            'vocab_size': len(self.vocab),
            'num_labels': len(self.label_manager),
            'labels': self.label_manager.get_labels(),
            'threshold': self.threshold,
            'device': str(self.device)
        }
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        vocab_path: str,
        label_path: str,
        config: Config = None,
        device: str = None
    ):
        """
        从检查点加载预测器
        
        Args:
            checkpoint_path: 模型检查点路径
            vocab_path: 词表路径
            label_path: 标签配置路径
            config: 配置对象
            device: 设备
        
        Returns:
            Predictor 实例
        """
        # 加载词表和标签
        vocab = Vocabulary.load(vocab_path)
        label_manager = LabelManager.load(label_path)
        
        # 加载配置
        if config is None:
            config = Config()
        
        # 创建模型
        model = TextCNN(
            vocab_size=len(vocab),
            embed_dim=config.model.embed_dim,
            num_classes=len(label_manager),
            filter_sizes=config.model.filter_sizes,
            num_filters=config.model.num_filters,
            dropout=config.model.dropout
        )
        
        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return cls(
            model=model,
            vocab=vocab,
            label_manager=label_manager,
            device=device,
            threshold=config.labels.threshold
        )
    
    def save(self, save_dir: str):
        """
        保存预测器相关文件
        
        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型
        model_path = os.path.join(save_dir, 'model.pt')
        torch.save(self.model.state_dict(), model_path)
        
        # 保存词表
        vocab_path = os.path.join(save_dir, 'vocab.pkl')
        self.vocab.save(vocab_path)
        
        # 保存标签配置
        label_path = os.path.join(save_dir, 'labels.pkl')
        self.label_manager.save(label_path)

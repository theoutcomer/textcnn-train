"""
数据增强模块
支持中文文本的数据增强技术
"""
import random
import jieba
from typing import List, Tuple
import synonyms  # 需要安装: pip install synonyms


class TextAugmenter:
    """
    文本数据增强器
    """
    
    def __init__(self, synonym_replace_ratio: float = 0.1, random_delete_ratio: float = 0.1):
        """
        初始化数据增强器
        
        Args:
            synonym_replace_ratio: 同义词替换比例
            random_delete_ratio: 随机删除比例
        """
        self.synonym_replace_ratio = synonym_replace_ratio
        self.random_delete_ratio = random_delete_ratio
    
    def synonym_replacement(self, text: str, n: int = None) -> str:
        """
        同义词替换
        随机选择 n 个词，替换为同义词
        
        Args:
            text: 原始文本
            n: 替换词数，默认为句子长度的 synonym_replace_ratio
        
        Returns:
            增强后的文本
        """
        words = list(jieba.cut(text))
        new_words = words.copy()
        
        # 确定替换数量
        if n is None:
            n = max(1, int(len(words) * self.synonym_replace_ratio))
        
        # 获取可替换的词（名词、动词、形容词）
        replaceable_indices = []
        for i, word in enumerate(words):
            if len(word) > 1:  # 只替换长度大于1的词
                replaceable_indices.append(i)
        
        if not replaceable_indices:
            return text
        
        # 随机选择要替换的词
        random.shuffle(replaceable_indices)
        num_replaced = 0
        
        for idx in replaceable_indices:
            if num_replaced >= n:
                break
            
            word = words[idx]
            synonyms_list = self._get_synonyms(word)
            
            if synonyms_list:
                # 选择同义词替换
                synonym = random.choice(synonyms_list)
                new_words[idx] = synonym
                num_replaced += 1
        
        return ''.join(new_words)
    
    def _get_synonyms(self, word: str) -> List[str]:
        """
        获取词的同义词
        
        Args:
            word: 输入词
        
        Returns:
            同义词列表
        """
        try:
            # 使用 synonyms 库获取同义词
            syns = synonyms.nearby(word)[0]
            # 过滤掉自己
            return [s for s in syns if s != word][:3]  # 最多返回3个
        except:
            return []
    
    def random_deletion(self, text: str, p: float = None) -> str:
        """
        随机删除词
        以概率 p 随机删除词
        
        Args:
            text: 原始文本
            p: 删除概率，默认使用 random_delete_ratio
        
        Returns:
            增强后的文本
        """
        if p is None:
            p = self.random_delete_ratio
        
        words = list(jieba.cut(text))
        
        if len(words) <= 1:
            return text
        
        new_words = []
        for word in words:
            # 以概率 p 删除词
            if random.random() > p:
                new_words.append(word)
        
        # 确保至少保留一个词
        if not new_words:
            new_words = [random.choice(words)]
        
        return ''.join(new_words)
    
    def random_swap(self, text: str, n: int = None) -> str:
        """
        随机交换词的位置
        
        Args:
            text: 原始文本
            n: 交换次数，默认为句子长度的 10%
        
        Returns:
            增强后的文本
        """
        words = list(jieba.cut(text))
        
        if len(words) < 2:
            return text
        
        if n is None:
            n = max(1, int(len(words) * 0.1))
        
        new_words = words.copy()
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        return ''.join(new_words)
    
    def random_insertion(self, text: str, n: int = None) -> str:
        """
        随机插入同义词
        
        Args:
            text: 原始文本
            n: 插入次数，默认为句子长度的 10%
        
        Returns:
            增强后的文本
        """
        words = list(jieba.cut(text))
        
        if n is None:
            n = max(1, int(len(words) * 0.1))
        
        new_words = words.copy()
        for _ in range(n):
            # 随机选择一个词
            word = random.choice(words)
            synonyms_list = self._get_synonyms(word)
            
            if synonyms_list:
                synonym = random.choice(synonyms_list)
                # 随机插入位置
                insert_idx = random.randint(0, len(new_words))
                new_words.insert(insert_idx, synonym)
        
        return ''.join(new_words)
    
    def augment(self, text: str, num_augments: int = 4) -> List[str]:
        """
        对单条文本进行多种增强
        
        Args:
            text: 原始文本
            num_augments: 增强样本数量
        
        Returns:
            增强后的文本列表
        """
        augmented_texts = []
        
        # 定义增强方法
        augment_methods = [
            self.synonym_replacement,
            self.random_deletion,
            self.random_swap,
            self.random_insertion
        ]
        
        # 随机选择增强方法
        for _ in range(num_augments):
            method = random.choice(augment_methods)
            augmented = method(text)
            if augmented != text:  # 只添加有变化的
                augmented_texts.append(augmented)
        
        return augmented_texts
    
    def augment_dataset(
        self,
        texts: List[str],
        labels: List[List[int]],
        augment_factor: float = 0.5
    ) -> Tuple[List[str], List[List[int]]]:
        """
        对整个数据集进行增强
        
        Args:
            texts: 文本列表
            labels: 标签列表
            augment_factor: 增强比例（相对于原数据集）
        
        Returns:
            增强后的文本和标签列表
        """
        augmented_texts = texts.copy()
        augmented_labels = labels.copy()
        
        # 确定要增强的样本数
        num_augment = int(len(texts) * augment_factor)
        
        # 随机选择要增强的样本
        indices = random.sample(range(len(texts)), num_augment)
        
        for idx in indices:
            text = texts[idx]
            label = labels[idx]
            
            # 生成增强样本
            aug_texts = self.augment(text, num_augments=2)
            
            for aug_text in aug_texts:
                augmented_texts.append(aug_text)
                augmented_labels.append(label)
        
        return augmented_texts, augmented_labels


class BackTranslationAugmenter:
    """
    回译数据增强（需要翻译API）
    中文 -> 英文 -> 中文
    """
    
    def __init__(self):
        # 这里可以集成翻译API，如百度翻译、Google翻译等
        self.enabled = False
    
    def augment(self, text: str) -> str:
        """
        回译增强
        需要接入翻译API才能实现
        """
        # 占位实现
        return text


# 简单的同义词词典（备用）
SYNONYM_DICT = {
    '好': ['优秀', '良好', '出色', '棒'],
    '大': ['巨大', '庞大', '很大'],
    '小': ['微小', '细小', '迷你'],
    '快': ['迅速', '快速', '飞快'],
    '慢': ['缓慢', '迟缓'],
    '新': ['新颖', '全新', '崭新'],
    '老': ['旧', '陈旧', '古老'],
    '高': ['很高', '高大', '高耸'],
    '低': ['低矮', '低下'],
    '多': ['很多', '许多', '大量'],
    '少': ['很少', '稀少', '少量'],
}

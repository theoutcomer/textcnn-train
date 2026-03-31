"""
数据预处理流水线
支持文本清洗、标准化、特征提取
"""
import re
import jieba
import jieba.posseg as pseg
from typing import List, Dict, Optional, Tuple
import html


class TextPreprocessor:
    """
    文本预处理器
    提供完整的文本清洗和标准化流程
    """
    
    def __init__(
        self,
        remove_html: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_phone: bool = False,
        remove_extra_spaces: bool = True,
        lowercase: bool = False,
        remove_punctuation: bool = False,
        remove_numbers: bool = False,
        remove_english: bool = False,
        min_text_length: int = 10,
        max_text_length: int = 2000
    ):
        """
        初始化预处理器
        
        Args:
            remove_html: 是否移除 HTML 标签
            remove_urls: 是否移除 URL
            remove_emails: 是否移除邮箱
            remove_phone: 是否移除电话号码
            remove_extra_spaces: 是否移除多余空格
            lowercase: 是否转为小写
            remove_punctuation: 是否移除标点
            remove_numbers: 是否移除数字
            remove_english: 是否移除英文
            min_text_length: 最小文本长度
            max_text_length: 最大文本长度
        """
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_phone = remove_phone
        self.remove_extra_spaces = remove_extra_spaces
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_english = remove_english
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        
        # 正则表达式模式
        self.url_pattern = re.compile(
            r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
            re.IGNORECASE
        )
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'1[3-9]\d{9}')
        self.html_pattern = re.compile(r'<[^>]+>')
        self.punctuation_pattern = re.compile(r'[^\w\s]')
        self.number_pattern = re.compile(r'\d+')
        self.english_pattern = re.compile(r'[a-zA-Z]+')
        self.space_pattern = re.compile(r'\s+')
    
    def clean(self, text: str) -> str:
        """
        清洗单条文本
        
        Args:
            text: 原始文本
        
        Returns:
            清洗后的文本
        """
        if not text or not isinstance(text, str):
            return ""
        
        # HTML 解码
        text = html.unescape(text)
        
        # 移除 HTML 标签
        if self.remove_html:
            text = self.html_pattern.sub('', text)
        
        # 移除 URL
        if self.remove_urls:
            text = self.url_pattern.sub('', text)
        
        # 移除邮箱
        if self.remove_emails:
            text = self.email_pattern.sub('', text)
        
        # 移除电话号码
        if self.remove_phone:
            text = self.phone_pattern.sub('', text)
        
        # 移除数字
        if self.remove_numbers:
            text = self.number_pattern.sub('', text)
        
        # 移除英文
        if self.remove_english:
            text = self.english_pattern.sub('', text)
        
        # 移除标点
        if self.remove_punctuation:
            text = self.punctuation_pattern.sub('', text)
        
        # 转为小写
        if self.lowercase:
            text = text.lower()
        
        # 移除多余空格
        if self.remove_extra_spaces:
            text = self.space_pattern.sub(' ', text)
        
        # 去除首尾空白
        text = text.strip()
        
        # 长度过滤
        if len(text) < self.min_text_length:
            return ""
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length]
        
        return text
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        批量清洗文本
        
        Args:
            texts: 文本列表
        
        Returns:
            清洗后的文本列表
        """
        return [self.clean(text) for text in texts]


class TextNormalizer:
    """
    文本标准化器
    处理同义词、停用词等
    """
    
    def __init__(
        self,
        stopwords_path: Optional[str] = None,
        synonym_dict: Optional[Dict[str, str]] = None
    ):
        """
        初始化标准化器
        
        Args:
            stopwords_path: 停用词文件路径
            synonym_dict: 同义词映射字典 {旧词: 标准词}
        """
        self.stopwords = set()
        self.synonym_dict = synonym_dict or {}
        
        # 加载停用词
        if stopwords_path and os.path.exists(stopwords_path):
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                self.stopwords = set(line.strip() for line in f if line.strip())
        else:
            # 默认停用词
            self.stopwords = {
                '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
                '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
                '你', '会', '着', '没有', '看', '好', '自己', '这', '那'
            }
    
    def normalize(self, text: str) -> str:
        """
        标准化文本
        
        Args:
            text: 输入文本
        
        Returns:
            标准化后的文本
        """
        words = list(jieba.cut(text))
        
        # 同义词替换
        words = [self.synonym_dict.get(w, w) for w in words]
        
        # 移除停用词
        words = [w for w in words if w not in self.stopwords and w.strip()]
        
        return ''.join(words)
    
    def extract_keywords(self, text: str, topk: int = 10) -> List[Tuple[str, str]]:
        """
        提取关键词（带词性）
        
        Args:
            text: 输入文本
            topk: 返回前 k 个关键词
        
        Returns:
            [(词, 词性), ...]
        """
        words = pseg.cut(text)
        
        # 过滤停用词和单字词
        keywords = [(w, f) for w, f in words 
                   if w not in self.stopwords and len(w) > 1]
        
        # 按词频排序
        word_freq = {}
        for w, f in keywords:
            word_freq[(w, f)] = word_freq.get((w, f), 0) + 1
        
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [item[0] for item in sorted_keywords[:topk]]


class DataPipeline:
    """
    数据预处理流水线
    整合清洗、标准化、分词等步骤
    """
    
    def __init__(
        self,
        preprocessor: Optional[TextPreprocessor] = None,
        normalizer: Optional[TextNormalizer] = None
    ):
        """
        初始化流水线
        
        Args:
            preprocessor: 预处理器
            normalizer: 标准化器
        """
        self.preprocessor = preprocessor or TextPreprocessor()
        self.normalizer = normalizer or TextNormalizer()
    
    def process(self, text: str, return_tokens: bool = False):
        """
        处理单条文本
        
        Args:
            text: 原始文本
            return_tokens: 是否返回分词结果
        
        Returns:
            处理后的文本，或 (文本, 分词列表)
        """
        # 清洗
        text = self.preprocessor.clean(text)
        
        if not text:
            return ("", []) if return_tokens else ""
        
        # 标准化
        text = self.normalizer.normalize(text)
        
        if return_tokens:
            tokens = list(jieba.cut(text))
            return text, tokens
        
        return text
    
    def process_batch(
        self,
        texts: List[str],
        return_tokens: bool = False
    ) -> List:
        """
        批量处理文本
        
        Args:
            texts: 文本列表
            return_tokens: 是否返回分词结果
        
        Returns:
            处理结果列表
        """
        results = []
        for text in texts:
            result = self.process(text, return_tokens)
            results.append(result)
        return results
    
    def build_vocabulary(
        self,
        texts: List[str],
        min_freq: int = 2,
        max_size: int = 50000
    ) -> Dict[str, int]:
        """
        从文本构建词表
        
        Args:
            texts: 文本列表
            min_freq: 最小词频
            max_size: 最大词表大小
        
        Returns:
            词表字典 {词: 索引}
        """
        word_freq = {}
        
        for text in texts:
            _, tokens = self.process(text, return_tokens=True)
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1
        
        # 过滤低频词
        filtered_words = {w: f for w, f in word_freq.items() if f >= min_freq}
        
        # 按词频排序
        sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)
        
        # 构建词表
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, _ in sorted_words[:max_size - 2]:
            vocab[word] = len(vocab)
        
        return vocab


import os


# 预设的预处理配置
PRESET_CONFIGS = {
    'default': TextPreprocessor(),
    'news': TextPreprocessor(
        remove_html=True,
        remove_urls=True,
        remove_emails=True,
        remove_extra_spaces=True,
        min_text_length=20,
        max_text_length=1000
    ),
    'social': TextPreprocessor(
        remove_html=True,
        remove_urls=False,  # 社交媒体保留 URL
        remove_emails=True,
        remove_extra_spaces=True,
        remove_punctuation=False,  # 保留表情符号等
        min_text_length=5,
        max_text_length=500
    ),
    'clean': TextPreprocessor(
        remove_html=True,
        remove_urls=True,
        remove_emails=True,
        remove_phone=True,
        remove_extra_spaces=True,
        remove_punctuation=True,
        remove_numbers=False,
        min_text_length=10,
        max_text_length=2000
    )
}


def get_preprocessor(preset: str = 'default', **kwargs) -> TextPreprocessor:
    """
    获取预设预处理器
    
    Args:
        preset: 预设名称 (default/news/social/clean)
        **kwargs: 覆盖配置参数
    
    Returns:
        预处理器实例
    """
    if preset in PRESET_CONFIGS:
        preprocessor = PRESET_CONFIGS[preset]
        # 更新参数
        for key, value in kwargs.items():
            if hasattr(preprocessor, key):
                setattr(preprocessor, key, value)
        return preprocessor
    else:
        return TextPreprocessor(**kwargs)

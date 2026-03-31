"""调试模型加载"""
import torch
import pickle
import jieba

# 加载检查点
checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu')
print('Checkpoint keys:', checkpoint.keys())
print('Best F1:', checkpoint.get('best_val_f1', 'N/A'))
print('Epoch:', checkpoint.get('epoch', 'N/A'))

# 加载词汇表
with open('checkpoints/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

print('\nVocab type:', type(vocab))
print('Vocab size:', len(vocab))

# 处理词汇表
if hasattr(vocab, 'word2idx'):
    vocab_dict = vocab.word2idx
else:
    vocab_dict = vocab

# 测试文本
text = "央行宣布降准降息，支持实体经济发展"
tokens = list(jieba.cut(text))
print(f'\nText: {text}')
print(f'Tokens: {tokens}')

print('\nToken IDs:')
for token in tokens[:10]:
    idx = vocab_dict.get(token, 'UNK')
    print(f'  {token}: {idx}')

# 检查特殊token
print('\nSpecial tokens:')
for token in ['<PAD>', '<UNK>']:
    idx = vocab_dict.get(token, 'NOT FOUND')
    print(f'  {token}: {idx}')

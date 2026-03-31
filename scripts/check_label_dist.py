"""检查标签分布"""
import json
from collections import Counter

with open('data/train.json', 'r', encoding='utf-8') as f:
    train = json.load(f)
with open('data/val.json', 'r', encoding='utf-8') as f:
    val = json.load(f)

print(f'训练集: {len(train)} 条')
print(f'验证集: {len(val)} 条')

all_labels = []
for item in train + val:
    all_labels.extend(item['labels'])

counts = Counter(all_labels)
print('\n标签分布:')
for label, count in counts.most_common():
    print(f'  {label}: {count}')

# 检查每篇文章的标签数
label_counts = [len(item['labels']) for item in train + val]
print(f'\n每篇文章平均标签数: {sum(label_counts)/len(label_counts):.2f}')
print(f'有标签的文章: {sum(1 for c in label_counts if c > 0)} / {len(label_counts)}')

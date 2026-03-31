"""测试不同阈值效果"""
import requests
import json

test_texts = [
    '央行宣布降准降息，支持实体经济发展',
    '粤港澳大湾区建设取得新进展',
    'A股三大指数集体上涨，科技股领涨',
    '美联储宣布加息25个基点',
    '新能源汽车销量创新高'
]

print('='*60)
print('测试不同阈值效果')
print('='*60)

for threshold in [0.1, 0.15, 0.2, 0.25]:
    print(f'\n阈值: {threshold}')
    print('-'*40)
    for text in test_texts[:3]:
        try:
            resp = requests.post('http://localhost:8082/predict', 
                               json={'text': text, 'threshold': threshold, 'return_probs': True},
                               timeout=5)
            result = resp.json()
            labels = result.get('labels', [])
            probs = result.get('probabilities', {})
            top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f'  文本: {text[:20]}...')
            print(f'  预测标签: {labels}')
            print(f'  Top3概率: {top3}')
        except Exception as e:
            print(f'  错误: {e}')

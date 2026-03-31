"""测试 API 服务"""
import requests
import json

url = 'http://localhost:8082/predict'

# 测试数据
test_cases = [
    {
        'text': '央行宣布降准降息，支持实体经济发展',
        'return_probs': True
    },
    {
        'text': '粤港澳大湾区建设取得新进展',
        'return_probs': True
    },
    {
        'text': 'A股三大指数集体上涨，科技股领涨',
        'return_probs': True
    }
]

print("=" * 60)
print("Testing TextCNN API Service")
print("=" * 60)

for i, data in enumerate(test_cases, 1):
    print(f"\nTest {i}:")
    print(f"Input: {data['text']}")
    
    try:
        response = requests.post(url, json=data)
        result = response.json()
        print(f"Predicted Labels: {result['labels']}")
        if 'probabilities' in result:
            probs = result['probabilities']
            top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"Top 3 Probabilities:")
            for label, prob in top3:
                print(f"  {label}: {prob:.4f}")
    except Exception as e:
        print(f"Error: {e}")

print("\n" + "=" * 60)
print("API Test Completed")
print("=" * 60)

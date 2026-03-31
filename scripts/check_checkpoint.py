"""检查模型检查点状态"""
import torch
import pickle

checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu')

print("=" * 50)
print("模型检查点状态")
print("=" * 50)
epoch = checkpoint.get('epoch', 'unknown')
best_f1 = checkpoint.get('best_f1', 'unknown')
print(f"已训练轮次: {epoch}")
print(f"最佳F1分数: {best_f1 if isinstance(best_f1, str) else f'{best_f1:.4f}'}")
print(f"标签数量: {len(checkpoint.get('labels', []))}")

# 加载标签
with open('checkpoints/labels.pkl', 'rb') as f:
    labels = pickle.load(f)
print(f"\n标签列表: {labels}")

# 检查是否有优化器状态
if 'optimizer_state_dict' in checkpoint:
    print("\n✓ 包含优化器状态，可以恢复训练")
else:
    print("\n✗ 不包含优化器状态，只能加载模型权重")

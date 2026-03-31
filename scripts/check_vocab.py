"""检查词汇表"""
import pickle

with open('checkpoints/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

print(f"Vocab type: {type(vocab)}")
print(f"Vocab size: {len(vocab)}")

if hasattr(vocab, 'word2idx'):
    print("Vocabulary object detected")
    print(f"Sample: {list(vocab.word2idx.items())[:5]}")
elif isinstance(vocab, dict):
    print("Dictionary detected")
    print(f"Sample: {list(vocab.items())[:5]}")

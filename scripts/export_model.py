"""导出模型为 ONNX 格式"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch
import pickle
from src.models.textcnn import TextCNN


def export_to_onnx(model, vocab, labels, output_path, max_len=512):
    """导出为 ONNX 格式"""
    model.eval()
    
    # 创建 dummy input
    dummy_input = torch.randint(0, len(vocab), (1, max_len))
    
    # 导出
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size'}
        },
        opset_version=11
    )
    print(f"ONNX model exported to: {output_path}")


def export_to_torchscript(model, output_path):
    """导出为 TorchScript 格式"""
    model.eval()
    scripted_model = torch.jit.script(model)
    scripted_model.save(output_path)
    print(f"TorchScript model exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Export TextCNN model')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing model files')
    parser.add_argument('--output_dir', type=str, default='exported_model',
                        help='Output directory')
    parser.add_argument('--format', type=str, default='onnx',
                        choices=['onnx', 'torchscript', 'both'],
                        help='Export format')
    
    args = parser.parse_args()
    
    # 加载模型
    checkpoint_path = os.path.join(args.model_dir, 'best_model.pt')
    vocab_path = os.path.join(args.model_dir, 'vocab.pkl')
    labels_path = os.path.join(args.model_dir, 'labels.pkl')
    
    print("Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    with open(labels_path, 'rb') as f:
        labels_data = pickle.load(f)
        labels = labels_data['idx2label']
    
    # 创建模型
    model_config = checkpoint['model_config']
    model = TextCNN(
        vocab_size=model_config['vocab_size'],
        embed_dim=model_config['embed_dim'],
        num_classes=model_config['num_classes'],
        filter_sizes=model_config.get('filter_sizes', [3, 4, 5]),
        num_filters=model_config.get('num_filters', 100),
        dropout=model_config.get('dropout', 0.5)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 导出模型
    if args.format in ['onnx', 'both']:
        onnx_path = os.path.join(args.output_dir, 'model.onnx')
        export_to_onnx(model, vocab, labels, onnx_path)
    
    if args.format in ['torchscript', 'both']:
        ts_path = os.path.join(args.output_dir, 'model.pt')
        export_to_torchscript(model, ts_path)
    
    # 保存词汇表和标签
    import json
    with open(os.path.join(args.output_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
        # vocab 可能是 Vocabulary 对象或 dict
        if hasattr(vocab, 'word2idx'):
            vocab_dict = vocab.word2idx
        else:
            vocab_dict = vocab
        json.dump(vocab_dict, f, ensure_ascii=False)
    
    with open(os.path.join(args.output_dir, 'labels.json'), 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False)
    
    # 保存模型配置
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(model_config, f)
    
    print(f"\nModel exported to: {args.output_dir}")
    print("Files:")
    for f in os.listdir(args.output_dir):
        print(f"  - {f}")


if __name__ == '__main__':
    main()

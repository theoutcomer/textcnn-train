"""
预测脚本
支持单条预测和批量预测
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import pandas as pd
from tqdm import tqdm

from src.predictor import Predictor
from src.utils.config import Config


def main():
    parser = argparse.ArgumentParser(description='Predict with TextCNN model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, required=True,
                        help='Path to vocabulary file')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to labels file')
    parser.add_argument('--input', type=str, required=True,
                        help='Input text or file path')
    parser.add_argument('--input_type', type=str, default='text',
                        choices=['text', 'file'],
                        help='Input type: text or file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Prediction threshold')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for file prediction')
    parser.add_argument('--return_probs', action='store_true',
                        help='Return probability scores')
    
    args = parser.parse_args()
    
    # 加载配置
    config = Config.from_yaml(args.config)
    
    print("=" * 50)
    print("TextCNN Prediction")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Vocab: {args.vocab}")
    print(f"Labels: {args.labels}")
    
    # 加载预测器
    print("\nLoading model...")
    predictor = Predictor.from_checkpoint(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        label_path=args.labels,
        config=config
    )
    
    # 更新阈值
    if args.threshold is not None:
        predictor.update_threshold(args.threshold)
    
    model_info = predictor.get_model_info()
    print(f"Model loaded successfully!")
    print(f"Labels: {model_info['labels']}")
    print(f"Threshold: {model_info['threshold']}")
    print(f"Device: {model_info['device']}")
    
    # 预测
    if args.input_type == 'text':
        # 单条预测
        print(f"\nInput text: {args.input}")
        result = predictor.predict(args.input, return_probs=args.return_probs)
        
        print("\nPrediction result:")
        print(f"  Labels: {result['labels']}")
        if args.return_probs:
            print("  Probabilities:")
            for label, prob in result['probabilities'].items():
                print(f"    {label}: {prob:.4f}")
    
    else:
        # 批量预测
        print(f"\nLoading input file: {args.input}")
        
        # 读取输入文件
        if args.input.endswith('.csv'):
            df = pd.read_csv(args.input)
            texts = df['text'].tolist()
        elif args.input.endswith('.json'):
            with open(args.input, 'r', encoding='utf-8') as f:
                data = json.load(f)
            texts = [item['text'] for item in data]
        else:
            with open(args.input, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        
        print(f"Total samples: {len(texts)}")
        
        # 批量预测
        print("\nPredicting...")
        results = predictor.predict_batch(
            texts,
            batch_size=args.batch_size,
            return_probs=args.return_probs
        )
        
        # 整理输出
        output_data = []
        for text, result in zip(texts, results):
            item = {
                'text': text,
                'predicted_labels': result['labels']
            }
            if args.return_probs:
                item['probabilities'] = result['probabilities']
            output_data.append(item)
        
        # 输出结果
        if args.output:
            # 保存到文件
            if args.output.endswith('.json'):
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
            elif args.output.endswith('.csv'):
                df_out = pd.DataFrame(output_data)
                df_out.to_csv(args.output, index=False)
            else:
                with open(args.output, 'w', encoding='utf-8') as f:
                    for item in output_data:
                        f.write(f"{item['text']}\t{','.join(item['predicted_labels'])}\n")
            
            print(f"\nResults saved to: {args.output}")
        else:
            # 打印到控制台
            print("\nPrediction results:")
            for i, item in enumerate(output_data[:10]):  # 只显示前10条
                print(f"\n[{i+1}] {item['text'][:50]}...")
                print(f"    Labels: {item['predicted_labels']}")
            
            if len(output_data) > 10:
                print(f"\n... and {len(output_data) - 10} more results")


if __name__ == '__main__':
    main()

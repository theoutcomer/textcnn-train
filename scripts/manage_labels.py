"""
标签管理工具
支持动态增删标签、更新配置
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import pickle
from typing import List

from src.data.dataset import LabelManager
from src.utils.config import Config


def load_label_manager(label_path: str) -> LabelManager:
    """加载标签管理器"""
    if os.path.exists(label_path):
        return LabelManager.load(label_path)
    return LabelManager()


def save_label_manager(label_manager: LabelManager, label_path: str):
    """保存标签管理器"""
    os.makedirs(os.path.dirname(label_path) or '.', exist_ok=True)
    label_manager.save(label_path)
    print(f"Labels saved to: {label_path}")


def add_labels(label_path: str, new_labels: List[str], config_path: str = None):
    """
    添加新标签
    
    Args:
        label_path: 标签文件路径
        new_labels: 新标签列表
        config_path: 配置文件路径（可选，同时更新配置）
    """
    label_manager = load_label_manager(label_path)
    
    print(f"Current labels ({len(label_manager)}): {label_manager.get_labels()}")
    
    added = []
    for label in new_labels:
        if label not in label_manager.label2idx:
            label_manager.add_labels([label])
            added.append(label)
            print(f"✓ Added label: {label}")
        else:
            print(f"✗ Label already exists: {label}")
    
    if added:
        save_label_manager(label_manager, label_path)
        print(f"\nUpdated labels ({len(label_manager)}): {label_manager.get_labels()}")
        
        # 更新配置文件
        if config_path and os.path.exists(config_path):
            config = Config.from_yaml(config_path)
            for label in added:
                config.labels.add_label(label)
            config.to_yaml(config_path)
            print(f"Configuration updated: {config_path}")
    else:
        print("\nNo new labels added")


def remove_labels(label_path: str, labels_to_remove: List[str], config_path: str = None):
    """
    移除标签
    
    Args:
        label_path: 标签文件路径
        labels_to_remove: 要移除的标签列表
        config_path: 配置文件路径（可选）
    """
    label_manager = load_label_manager(label_path)
    
    print(f"Current labels ({len(label_manager)}): {label_manager.get_labels()}")
    
    removed = []
    for label in labels_to_remove:
        if label in label_manager.label2idx:
            # 注意：LabelManager 目前不支持直接删除，这里只是从配置中移除
            removed.append(label)
            print(f"✓ Marked for removal: {label}")
        else:
            print(f"✗ Label not found: {label}")
    
    if removed:
        print("\n⚠️  Warning: Removing labels requires model retraining!")
        print("Labels marked for removal will be excluded from future training.")
        
        # 更新配置文件
        if config_path and os.path.exists(config_path):
            config = Config.from_yaml(config_path)
            for label in removed:
                config.labels.remove_label(label)
            config.to_yaml(config_path)
            print(f"Configuration updated: {config_path}")


def list_labels(label_path: str):
    """列出所有标签"""
    label_manager = load_label_manager(label_path)
    
    print(f"\nTotal labels: {len(label_manager)}")
    print("Labels:")
    for i, label in enumerate(label_manager.get_labels()):
        print(f"  {i}: {label}")


def export_labels(label_path: str, output_path: str, format: str = 'json'):
    """
    导出标签配置
    
    Args:
        label_path: 标签文件路径
        output_path: 输出路径
        format: 输出格式 (json/csv/txt)
    """
    label_manager = load_label_manager(label_path)
    labels = label_manager.get_labels()
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    if format == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'labels': labels,
                'count': len(labels),
                'label2idx': label_manager.label2idx
            }, f, ensure_ascii=False, indent=2)
    
    elif format == 'csv':
        import pandas as pd
        df = pd.DataFrame({
            'index': range(len(labels)),
            'label': labels
        })
        df.to_csv(output_path, index=False, encoding='utf-8')
    
    elif format == 'txt':
        with open(output_path, 'w', encoding='utf-8') as f:
            for label in labels:
                f.write(f"{label}\n")
    
    print(f"Labels exported to: {output_path} ({format})")


def import_labels(input_path: str, label_path: str, config_path: str = None):
    """
    从文件导入标签
    
    Args:
        input_path: 输入文件路径
        label_path: 标签文件保存路径
        config_path: 配置文件路径（可选）
    """
    # 读取标签
    if input_path.endswith('.json'):
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        labels = data.get('labels', [])
    
    elif input_path.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(input_path)
        labels = df['label'].tolist()
    
    elif input_path.endswith('.txt'):
        with open(input_path, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f if line.strip()]
    
    else:
        raise ValueError(f"Unsupported file format: {input_path}")
    
    print(f"Importing {len(labels)} labels from: {input_path}")
    
    # 创建新的标签管理器
    label_manager = LabelManager()
    label_manager.add_labels(labels)
    save_label_manager(label_manager, label_path)
    
    # 更新配置
    if config_path:
        config = Config()
        config.labels.labels = labels
        config.to_yaml(config_path)
        print(f"Configuration created: {config_path}")


def main():
    parser = argparse.ArgumentParser(description='Label management tool')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # 添加标签
    add_parser = subparsers.add_parser('add', help='Add new labels')
    add_parser.add_argument('--labels', type=str, required=True,
                           help='Comma-separated labels to add')
    add_parser.add_argument('--label_path', type=str, default='checkpoints/labels.pkl',
                           help='Path to labels file')
    add_parser.add_argument('--config', type=str, default='configs/config.yaml',
                           help='Path to config file')
    
    # 移除标签
    remove_parser = subparsers.add_parser('remove', help='Remove labels')
    remove_parser.add_argument('--labels', type=str, required=True,
                              help='Comma-separated labels to remove')
    remove_parser.add_argument('--label_path', type=str, default='checkpoints/labels.pkl',
                              help='Path to labels file')
    remove_parser.add_argument('--config', type=str, default='configs/config.yaml',
                              help='Path to config file')
    
    # 列出标签
    list_parser = subparsers.add_parser('list', help='List all labels')
    list_parser.add_argument('--label_path', type=str, default='checkpoints/labels.pkl',
                            help='Path to labels file')
    
    # 导出标签
    export_parser = subparsers.add_parser('export', help='Export labels')
    export_parser.add_argument('--label_path', type=str, default='checkpoints/labels.pkl',
                              help='Path to labels file')
    export_parser.add_argument('--output', type=str, required=True,
                              help='Output file path')
    export_parser.add_argument('--format', type=str, default='json',
                              choices=['json', 'csv', 'txt'],
                              help='Output format')
    
    # 导入标签
    import_parser = subparsers.add_parser('import', help='Import labels from file')
    import_parser.add_argument('--input', type=str, required=True,
                              help='Input file path')
    import_parser.add_argument('--label_path', type=str, default='checkpoints/labels.pkl',
                              help='Path to save labels file')
    import_parser.add_argument('--config', type=str, default='configs/config.yaml',
                              help='Path to config file')
    
    args = parser.parse_args()
    
    if args.command == 'add':
        new_labels = [l.strip() for l in args.labels.split(',')]
        add_labels(args.label_path, new_labels, args.config)
    
    elif args.command == 'remove':
        labels_to_remove = [l.strip() for l in args.labels.split(',')]
        remove_labels(args.label_path, labels_to_remove, args.config)
    
    elif args.command == 'list':
        list_labels(args.label_path)
    
    elif args.command == 'export':
        export_labels(args.label_path, args.output, args.format)
    
    elif args.command == 'import':
        import_labels(args.input, args.label_path, args.config)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

"""从数据库导出训练数据"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import pymysql
import jieba
from collections import Counter
from typing import List, Dict

# 数据库配置
DB_CONFIG = {
    'host': '110.43.247.31',
    'port': 28550,
    'database': 'cbcms',
    'user': '21xmt_user',
    'password': 'CMScD!T7qCW%f1moUyw2F!AMzCAR0m',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}


def get_training_data(min_content_length: int = 50) -> List[Dict]:
    """从数据库获取带标签的训练数据"""
    conn = pymysql.connect(**DB_CONFIG)
    
    try:
        with conn.cursor() as cursor:
            # 查询带标签的文章
            cursor.execute("""
                SELECT 
                    a.id,
                    a.title,
                    a.content,
                    a.papername,
                    GROUP_CONCAT(DISTINCT s.catname) as labels
                FROM cnfanews_articles a
                JOIN article_label_mapping al ON a.id = al.article_id
                JOIN subcat_morning s ON al.subcat_id = s.subcat_id
                WHERE LENGTH(a.content) >= %s
                GROUP BY a.id
                ORDER BY a.created_at DESC
            """, (min_content_length,))
            
            results = cursor.fetchall()
            
            # 转换为训练格式
            training_data = []
            for row in results:
                # 合并标题和内容
                text = f"{row['title']} {row['content']}".strip()
                # 分割标签
                labels = [l.strip() for l in row['labels'].split(',') if l.strip()]
                
                training_data.append({
                    'text': text,
                    'labels': labels,
                    'source': row['papername']
                })
            
            return training_data
    finally:
        conn.close()


def analyze_data(data: List[Dict]):
    """分析数据分布"""
    print("="*60)
    print("训练数据分析")
    print("="*60)
    print(f"总样本数: {len(data)}")
    
    # 标签统计
    label_counter = Counter()
    for item in data:
        for label in item['labels']:
            label_counter[label] += 1
    
    print(f"\n标签分布:")
    for label, count in label_counter.most_common():
        print(f"  {label}: {count} ({count/len(data)*100:.1f}%)")
    
    # 文本长度统计
    text_lengths = [len(item['text']) for item in data]
    print(f"\n文本长度统计:")
    print(f"  平均: {sum(text_lengths)/len(text_lengths):.0f} 字符")
    print(f"  最短: {min(text_lengths)} 字符")
    print(f"  最长: {max(text_lengths)} 字符")
    
    # 每篇文章平均标签数
    avg_labels = sum(len(item['labels']) for item in data) / len(data)
    print(f"\n平均每篇文章标签数: {avg_labels:.2f}")


def export_to_json(data: List[Dict], output_path: str):
    """导出为 JSON 格式"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 已导出到: {output_path}")


def main():
    print("从数据库导出训练数据...")
    
    # 获取数据
    data = get_training_data(min_content_length=50)
    
    if not data:
        print("❌ 数据库中没有训练数据")
        print("请先运行: python scripts/fetch_and_store_cnfanews.py")
        return
    
    # 分析数据
    analyze_data(data)
    
    # 导出
    output_path = 'data/cnfanews_training.json'
    os.makedirs('data', exist_ok=True)
    export_to_json(data, output_path)
    
    # 同时导出标签列表
    all_labels = set()
    for item in data:
        all_labels.update(item['labels'])
    
    with open('data/labels.txt', 'w', encoding='utf-8') as f:
        for label in sorted(all_labels):
            f.write(f"{label}\n")
    print(f"✓ 标签列表已导出到: data/labels.txt")


if __name__ == '__main__':
    main()

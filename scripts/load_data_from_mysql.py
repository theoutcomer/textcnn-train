"""
从 MySQL 数据库加载数据
用于 TextCNN 训练
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pymysql
import pandas as pd
import json
import argparse
from typing import List, Dict, Tuple


def connect_db(
    host: str = '110.43.247.31',
    port: int = 28550,
    database: str = 'cbcms',
    user: str = '21xmt_user',
    password: str = 'CMScD!T7qCW%f1moUyw2F!AMzCAR0m'
):
    """
    连接 MySQL 数据库
    """
    conn = pymysql.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    return conn


def get_table_structure(conn, table_name: str) -> List[Dict]:
    """
    获取表结构
    """
    with conn.cursor() as cursor:
        cursor.execute(f"DESCRIBE {table_name}")
        result = cursor.fetchall()
    return result


def get_sample_data(conn, table_name: str, limit: int = 5) -> List[Dict]:
    """
    获取样本数据
    """
    with conn.cursor() as cursor:
        cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
        result = cursor.fetchall()
    return result


def get_all_tables(conn) -> List[str]:
    """
    获取所有表名
    """
    with conn.cursor() as cursor:
        cursor.execute("SHOW TABLES")
        result = cursor.fetchall()
    # 提取表名（字典的第一个值）
    tables = [list(row.values())[0] for row in result]
    return tables


def explore_database():
    """
    探索数据库结构
    """
    conn = connect_db()
    
    try:
        print("=" * 60)
        print("数据库探索")
        print("=" * 60)
        
        # 获取所有表
        tables = get_all_tables(conn)
        print(f"\n数据库共有 {len(tables)} 个表:")
        for i, table in enumerate(tables, 1):
            print(f"  {i}. {table}")
        
        # 查看 subcat_morning 表结构
        if 'subcat_morning' in tables:
            print("\n" + "=" * 60)
            print("subcat_morning 表结构（标签表）")
            print("=" * 60)
            structure = get_table_structure(conn, 'subcat_morning')
            for field in structure:
                print(f"  {field['Field']}: {field['Type']} (Null: {field['Null']}, Key: {field['Key']})")
            
            # 查看样本数据
            print("\n" + "-" * 60)
            print("subcat_morning 样本数据:")
            print("-" * 60)
            samples = get_sample_data(conn, 'subcat_morning', limit=5)
            for i, sample in enumerate(samples, 1):
                print(f"\n  记录 {i}:")
                for key, value in sample.items():
                    print(f"    {key}: {value}")
        
        # 查找可能包含文章内容的表
        print("\n" + "=" * 60)
        print("可能的文章数据表:")
        print("=" * 60)
        article_keywords = ['article', 'post', 'content', 'news', 'text', 'blog', 'doc']
        for table in tables:
            if any(keyword in table.lower() for keyword in article_keywords):
                print(f"\n  表名: {table}")
                structure = get_table_structure(conn, table)
                print(f"  字段:")
                for field in structure:
                    print(f"    - {field['Field']}: {field['Type']}")
                
                # 显示样本
                print(f"  样本数据:")
                samples = get_sample_data(conn, table, limit=2)
                for sample in samples:
                    print(f"    {sample}")
    
    finally:
        conn.close()


def load_training_data(
    article_table: str,
    text_column: str,
    label_column: str = None,
    label_table: str = None,
    join_key: str = None,
    output_path: str = 'data/train_from_db.json',
    limit: int = None
):
    """
    从数据库加载训练数据
    
    Args:
        article_table: 文章表名
        text_column: 文本内容字段名
        label_column: 标签字段名（如果在同一表）
        label_table: 标签表名（如果单独存储）
        join_key: 关联字段
        output_path: 输出文件路径
        limit: 限制数量
    """
    conn = connect_db()
    
    try:
        print("=" * 60)
        print("加载训练数据")
        print("=" * 60)
        
        # 构建查询
        if label_table and join_key:
            # 多表关联查询
            query = f"""
                SELECT a.{text_column} as text, GROUP_CONCAT(l.name) as labels
                FROM {article_table} a
                LEFT JOIN {label_table} l ON a.{join_key} = l.{join_key}
                WHERE a.{text_column} IS NOT NULL AND a.{text_column} != ''
                GROUP BY a.id
            """
        elif label_column:
            # 单表查询
            query = f"""
                SELECT {text_column} as text, {label_column} as labels
                FROM {article_table}
                WHERE {text_column} IS NOT NULL AND {text_column} != ''
            """
        else:
            # 只查询文本
            query = f"""
                SELECT {text_column} as text
                FROM {article_table}
                WHERE {text_column} IS NOT NULL AND {text_column} != ''
            """
        
        if limit:
            query += f" LIMIT {limit}"
        
        print(f"\n执行查询:\n{query}")
        
        # 执行查询
        df = pd.read_sql(query, conn)
        
        print(f"\n共加载 {len(df)} 条数据")
        print(f"\n数据预览:")
        print(df.head())
        
        # 转换为 JSON 格式
        data = []
        for _, row in df.iterrows():
            item = {'text': row['text']}
            if 'labels' in row and pd.notna(row['labels']):
                # 处理标签（可能是逗号分隔的字符串）
                labels = str(row['labels']).split(',') if ',' in str(row['labels']) else [str(row['labels'])]
                item['labels'] = [l.strip() for l in labels]
            else:
                item['labels'] = []
            data.append(item)
        
        # 保存
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n数据已保存到: {output_path}")
        
        # 统计标签
        if data and data[0].get('labels'):
            from collections import Counter
            all_labels = []
            for item in data:
                all_labels.extend(item['labels'])
            label_counts = Counter(all_labels)
            
            print(f"\n标签分布:")
            for label, count in label_counts.most_common():
                print(f"  {label}: {count}")
        
        return data
    
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description='Load data from MySQL for TextCNN')
    parser.add_argument('--explore', action='store_true',
                        help='探索数据库结构')
    parser.add_argument('--article_table', type=str,
                        help='文章表名')
    parser.add_argument('--text_column', type=str, default='content',
                        help='文本内容字段名')
    parser.add_argument('--label_column', type=str,
                        help='标签字段名（单表）')
    parser.add_argument('--label_table', type=str,
                        help='标签表名（单独存储）')
    parser.add_argument('--join_key', type=str,
                        help='关联字段')
    parser.add_argument('--output', type=str, default='data/train_from_db.json',
                        help='输出文件路径')
    parser.add_argument('--limit', type=int,
                        help='限制加载数量')
    
    args = parser.parse_args()
    
    if args.explore:
        explore_database()
    elif args.article_table:
        load_training_data(
            article_table=args.article_table,
            text_column=args.text_column,
            label_column=args.label_column,
            label_table=args.label_table,
            join_key=args.join_key,
            output_path=args.output,
            limit=args.limit
        )
    else:
        parser.print_help()
        print("\n建议先运行: python load_data_from_mysql.py --explore")


if __name__ == '__main__':
    main()

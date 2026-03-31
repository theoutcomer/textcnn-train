"""从数据库导出数据并划分训练集和验证集"""
import pymysql
import json
import random
from sklearn.model_selection import train_test_split

def export_and_split(test_size=0.2, random_state=42):
    """从数据库导出数据并划分训练/验证集"""
    conn = pymysql.connect(
        host='110.43.247.31', port=28550, database='cbcms',
        user='21xmt_user', password='CMScD!T7qCW%f1moUyw2F!AMzCAR0m',
        charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor
    )
    
    with conn.cursor() as cursor:
        # 查询已打标签的文章
        cursor.execute("""
            SELECT 
                a.title,
                a.content as text,
                GROUP_CONCAT(sm.catname) as labels
            FROM cnfanews_articles a
            JOIN article_label_mapping alm ON a.id = alm.article_id
            JOIN subcat_morning sm ON alm.subcat_id = sm.subcat_id
            WHERE LENGTH(a.content) > 100
            GROUP BY a.id
            HAVING COUNT(*) > 0
        """)
        
        results = cursor.fetchall()
    conn.close()
    
    # 转换为训练格式
    data = []
    for row in results:
        labels = row['labels'].split(',') if row['labels'] else []
        data.append({
            'title': row['title'],
            'text': row['text'],
            'labels': labels
        })
    
    print(f"从数据库导出 {len(data)} 条数据")
    
    # 划分训练集和验证集
    train_data, val_data = train_test_split(
        data, test_size=test_size, random_state=random_state
    )
    
    # 保存
    with open('data/train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open('data/val.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")
    
    return len(train_data), len(val_data)

if __name__ == '__main__':
    export_and_split()

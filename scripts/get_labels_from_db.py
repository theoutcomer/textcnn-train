"""从数据库获取标签并更新配置"""
import pymysql
import yaml

def get_labels():
    conn = pymysql.connect(
        host='110.43.247.31', port=28550, database='cbcms',
        user='21xmt_user', password='CMScD!T7qCW%f1moUyw2F!AMzCAR0m',
        charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor
    )
    with conn.cursor() as cursor:
        cursor.execute('SELECT catname FROM subcat_morning WHERE is_morning=1 ORDER BY subcat_id')
        labels = [row['catname'] for row in cursor.fetchall()]
    conn.close()
    return labels

if __name__ == '__main__':
    labels = get_labels()
    print(f"从数据库获取到 {len(labels)} 个标签:")
    for i, label in enumerate(labels, 1):
        print(f"  {i}. {label}")

"""查看 ods_fanwen_article 表结构"""
import pymysql

conn = pymysql.connect(
    host='mysql-internet-cn-south-1-5110e80fbe8c44c7.rds.jdcloud.com', port=3306,
    database='sfc_mda', user='qinzongqun', password='48HMeJDudJVC8TE',
    charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor
)

with conn.cursor() as cursor:
    # 查看表结构
    cursor.execute('DESCRIBE ods_fanwen_article')
    columns = cursor.fetchall()
    
    print("表名: ods_fanwen_article")
    print("=" * 60)
    for col in columns:
        print(f"  {col['Field']}: {col['Type']}")
    
    # 查看记录数
    cursor.execute('SELECT COUNT(*) as cnt FROM ods_fanwen_article')
    count = cursor.fetchone()['cnt']
    print(f"\n记录数: {count:,}")
    
    # 查看样本数据
    if count > 0:
        cursor.execute('SELECT * FROM ods_fanwen_article LIMIT 1')
        sample = cursor.fetchone()
        print(f"\n样本数据:")
        for key, value in sample.items():
            if value and len(str(value)) > 200:
                value = str(value)[:200] + "..."
            print(f"  {key}: {value}")

conn.close()

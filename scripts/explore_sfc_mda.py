"""探索 sfc_mda 数据库结构"""
import pymysql

def explore_database():
    conn = pymysql.connect(
        host='mysql-internet-cn-south-1-5110e80fbe8c44c7.rds.jdcloud.com',
        port=3306,
        database='sfc_mda',
        user='qinzongqun',
        password='48HMeJDudJVC8TE',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    
    try:
        with conn.cursor() as cursor:
            # 获取所有表
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            
            print("=" * 60)
            print("sfc_mda 数据库表结构")
            print("=" * 60)
            
            for table_info in tables:
                table_name = list(table_info.values())[0]
                print(f"\n表名: {table_name}")
                print("-" * 40)
                
                # 获取表结构
                cursor.execute(f"DESCRIBE {table_name}")
                columns = cursor.fetchall()
                
                for col in columns:
                    print(f"  {col['Field']}: {col['Type']}")
                
                # 获取表记录数
                cursor.execute(f"SELECT COUNT(*) as cnt FROM {table_name}")
                count = cursor.fetchone()['cnt']
                print(f"  记录数: {count}")
                
                # 显示前3条数据样本
                if count > 0:
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
                    sample = cursor.fetchone()
                    print(f"  样本数据:")
                    for key, value in sample.items():
                        if value and len(str(value)) > 100:
                            value = str(value)[:100] + "..."
                        print(f"    {key}: {value}")
    
    finally:
        conn.close()

if __name__ == '__main__':
    explore_database()

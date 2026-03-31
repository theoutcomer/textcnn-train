"""查看 sfc_mda 数据库统计"""
import pymysql

conn = pymysql.connect(
    host='mysql-internet-cn-south-1-5110e80fbe8c44c7.rds.jdcloud.com', port=3306,
    database='sfc_mda', user='qinzongqun', password='48HMeJDudJVC8TE',
    charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor
)

with conn.cursor() as cursor:
    # 查看文章表总数
    cursor.execute('SELECT COUNT(*) as cnt FROM ads_media_article')
    total = cursor.fetchone()['cnt']
    print(f'ads_media_article 总记录数: {total:,}')
    
    # 查看有内容的记录
    cursor.execute('SELECT COUNT(*) as cnt FROM ads_media_article_content')
    content_total = cursor.fetchone()['cnt']
    print(f'ads_media_article_content 总记录数: {content_total:,}')
    
    # 查看2023年前的文章
    cursor.execute('SELECT COUNT(*) as cnt FROM ads_media_article_2023_before')
    old_total = cursor.fetchone()['cnt']
    print(f'ads_media_article_2023_before 总记录数: {old_total:,}')
    
    # 查看一条有内容的文章
    cursor.execute('''
        SELECT a.title, c.content 
        FROM ads_media_article a
        JOIN ads_media_article_content c ON a.id = c.id
        WHERE LENGTH(c.content) > 100
        LIMIT 1
    ''')
    result = cursor.fetchone()
    if result:
        print(f'\n样本文章:')
        print(f'标题: {result["title"]}')
        print(f'内容前200字: {result["content"][:200]}...')

conn.close()

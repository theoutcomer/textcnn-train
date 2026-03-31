"""检查数据统计"""
import pymysql

conn = pymysql.connect(
    host='110.43.247.31', port=28550, database='cbcms',
    user='21xmt_user', password='CMScD!T7qCW%f1moUyw2F!AMzCAR0m',
    charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor
)

with conn.cursor() as cursor:
    cursor.execute('SELECT COUNT(*) as cnt FROM cnfanews_articles')
    articles = cursor.fetchone()['cnt']
    cursor.execute('SELECT COUNT(*) as cnt FROM article_label_mapping')
    labels = cursor.fetchone()['cnt']
    print(f'文章总数: {articles}')
    print(f'标签关联: {labels}')

conn.close()

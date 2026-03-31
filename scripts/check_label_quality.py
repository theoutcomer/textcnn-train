"""检查标签质量"""
import pymysql

conn = pymysql.connect(
    host='110.43.247.31', port=28550, database='cbcms',
    user='21xmt_user', password='CMScD!T7qCW%f1moUyw2F!AMzCAR0m',
    charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor
)

with conn.cursor() as cursor:
    # 标签分布
    cursor.execute("""
        SELECT 
            sm.catname,
            COUNT(*) as article_count
        FROM article_label_mapping alm
        JOIN subcat_morning sm ON alm.subcat_id = sm.subcat_id
        GROUP BY sm.subcat_id, sm.catname
        ORDER BY article_count DESC
    """)
    tags = cursor.fetchall()
    
    print("标签分布:")
    for tag in tags:
        print(f"  {tag['catname']}: {tag['article_count']} 篇")
    
    # 单标签文章（可疑）
    cursor.execute("""
        SELECT 
            a.id,
            a.title,
            sm.catname as single_label
        FROM cnfanews_articles a
        JOIN article_label_mapping alm ON a.id = alm.article_id
        JOIN subcat_morning sm ON alm.subcat_id = sm.subcat_id
        WHERE a.id IN (
            SELECT article_id
            FROM article_label_mapping
            GROUP BY article_id
            HAVING COUNT(*) = 1
        )
        AND sm.catname IN ('公司', '全球', '投资')
        LIMIT 5
    """)
    suspicious = cursor.fetchall()
    
    print(f"\n可疑单标签文章（仅显示前5）:")
    for item in suspicious:
        print(f"  [{item['id']}] {item['title'][:40]}... -> {item['single_label']}")

conn.close()

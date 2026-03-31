"""从 sfc_mda 数据库导入文章数据用于训练"""
import pymysql
import json
import re
from tqdm import tqdm


def clean_html(html_content):
    """清理 HTML 标签"""
    if not html_content:
        return ""
    text = re.sub(r'<[^>]+>', '', html_content)
    text = text.replace('&nbsp;', ' ').replace('&quot;', '"').replace('&amp;', '&')
    text = text.replace('&lt;', '<').replace('&gt;', '>')
    text = ' '.join(text.split())
    return text.strip()


def get_sfc_connection():
    """连接 sfc_mda 数据库"""
    return pymysql.connect(
        host='mysql-internet-cn-south-1-5110e80fbe8c44c7.rds.jdcloud.com',
        port=3306,
        database='sfc_mda',
        user='qinzongqun',
        password='48HMeJDudJVC8TE',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )


def get_cbcms_connection():
    """连接 cbcms 数据库"""
    return pymysql.connect(
        host='110.43.247.31', port=28550, database='cbcms',
        user='21xmt_user', password='CMScD!T7qCW%f1moUyw2F!AMzCAR0m',
        charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor
    )


def auto_label_article(title, content, labels_info):
    """基于规则自动匹配标签"""
    matched_labels = []
    text = (title + ' ' + content).lower()
    title_lower = title.lower()
    
    # 标签关键词映射
    keyword_mapping = {
        '粤港澳': ['粤港澳', '大湾区', '香港', '澳门', '深圳', '广州', '珠三角', '前海', '横琴', '南沙'],
        '学习经济': ['学习', '习近平', '讲话', '政策解读', '会议精神', '重要讲话', '指示批示'],
        '宏观': ['宏观', '经济政策', '货币政策', '财政政策', 'GDP', '通胀', 'CPI', '央行', '人民银行', '发改委'],
        '金融': ['金融', '银行', '保险', '信贷', '利率', '汇率', '融资', '贷款', '存款', '信托'],
        '证券': ['证券', '股票', '股市', 'A股', '港股', '美股', '指数', '涨停', '跌停', '牛市', '熊市', '券商'],
        '全球': ['全球', '国际', '美国', '欧洲', '日本', '美联储', '华尔街', '海外市场', '国际贸易'],
        '评论': ['评论', '分析', '观点', '解读', '展望', '研判', '综述', '时评', '社评'],
        '投资': ['投资', '理财', '基金', '私募', '公募', '资产配置', '收益率', '回报', '风投'],
        '视频': ['视频', '直播', '访谈', '对话', '节目', '视频号', '短视频'],
        '公司': ['公司', '企业', '上市公司', '财报', '业绩', '年报', '季报', 'IPO', '并购', '重组']
    }
    
    scarce_labels = ['粤港澳', '学习经济', '宏观']
    
    for label in labels_info:
        catname = label['catname']
        if not catname:
            continue
        
        is_scarce = catname in scarce_labels
        keywords = keyword_mapping.get(catname, [])
        
        if is_scarce:
            if any(kw in title_lower for kw in keywords):
                matched_labels.append(label['subcat_id'])
                continue
            if any(kw in text for kw in keywords):
                matched_labels.append(label['subcat_id'])
                continue
        else:
            if catname.lower() in text:
                matched_labels.append(label['subcat_id'])
                continue
            match_count = sum(1 for kw in keywords if kw in text)
            if match_count >= 2 or any(kw in title_lower for kw in keywords):
                matched_labels.append(label['subcat_id'])
    
    return matched_labels


def import_articles(limit=5000):
    """从 sfc_mda 导入文章到 cbcms"""
    print("=" * 60)
    print(f"从 sfc_mda 导入文章 (limit: {limit})")
    print("=" * 60)
    
    # 连接两个数据库
    sfc_conn = get_sfc_connection()
    cbcms_conn = get_cbcms_connection()
    
    try:
        # 获取 cbcms 标签信息
        with cbcms_conn.cursor() as cursor:
            cursor.execute('SELECT subcat_id, catname FROM subcat_morning WHERE is_morning=1')
            labels_info = cursor.fetchall()
        
        print(f"\n从 cbcms 获取到 {len(labels_info)} 个标签")
        
        # 从 sfc_mda 获取文章
        with sfc_conn.cursor() as cursor:
            print("\n正在从 sfc_mda 查询文章...")
            cursor.execute('''
                SELECT a.id, a.title, c.content, a.platform_name, a.account_name
                FROM ads_media_article a
                JOIN ads_media_article_content c ON a.id = c.id
                WHERE LENGTH(c.content) > 200
                ORDER BY RAND()
                LIMIT %s
            ''', (limit,))
            articles = cursor.fetchall()
        
        print(f"从 sfc_mda 获取到 {len(articles)} 篇文章")
        
        # 导入到 cbcms
        imported = 0
        labeled = 0
        
        with cbcms_conn.cursor() as cursor:
            for article in tqdm(articles, desc="导入文章"):
                try:
                    # 清理内容
                    content = clean_html(article['content'])
                    if len(content) < 100:
                        continue
                    
                    # 检查是否已存在
                    cursor.execute(
                        "SELECT id FROM cnfanews_articles WHERE articlesequenceid = %s",
                        (article['id'],)
                    )
                    if cursor.fetchone():
                        continue
                    
                    # 插入文章
                    cursor.execute('''
                        INSERT INTO cnfanews_articles 
                        (articlesequenceid, title, content, papername, api_typename, source_url)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    ''', (
                        article['id'],
                        article['title'],
                        content,
                        article.get('account_name', ''),
                        article.get('platform_name', ''),
                        ''
                    ))
                    
                    article_db_id = cursor.lastrowid
                    imported += 1
                    
                    # 自动打标签
                    matched_labels = auto_label_article(
                        article['title'], content, labels_info
                    )
                    
                    for subcat_id in matched_labels:
                        cursor.execute('''
                            INSERT INTO article_label_mapping (article_id, subcat_id, confidence)
                            VALUES (%s, %s, %s)
                            ON DUPLICATE KEY UPDATE confidence = VALUES(confidence)
                        ''', (article_db_id, subcat_id, 1.0))
                        labeled += 1
                    
                    if imported % 100 == 0:
                        cbcms_conn.commit()
                
                except Exception as e:
                    print(f"处理文章失败: {e}")
                    continue
            
            cbcms_conn.commit()
        
        print(f"\n{'=' * 60}")
        print(f"导入完成!")
        print(f"新增文章: {imported} 篇")
        print(f"标签关联: {labeled} 个")
        print(f"{'=' * 60}")
        
    finally:
        sfc_conn.close()
        cbcms_conn.close()


if __name__ == '__main__':
    import_articles(limit=5000)

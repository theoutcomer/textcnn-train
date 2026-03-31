"""
从 Cnfanews API 获取文章并存储到 MySQL 数据库
建立与 subcat_morning 标签表的关联
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import requests
import json
import hashlib
import time
import argparse
import re
from typing import List, Dict, Tuple, Optional
import pymysql
from datetime import datetime
from tqdm import tqdm


class CnfanewsAPI:
    """Cnfanews API 客户端"""
    
    def __init__(self):
        self.appid = "21shiji_autoget"
        self.password = "Ttle57axXeH2Uwcp0sSw9w=="
        self.base_url = "http://api.cnfanews.com:8000/clientdata_rti.ashx"
    
    def _generate_sign(self, *params) -> str:
        """生成签名 MD5 大写"""
        sign_str = ''.join(str(p) for p in params)
        return hashlib.md5(sign_str.encode()).hexdigest().upper()
    
    def get_website_list(self) -> List[Dict]:
        """获取站点列表"""
        timestamp = int(time.time())
        what_do = "GetSourceList"
        sign = self._generate_sign(self.appid, timestamp, what_do, self.password)
        
        params = {
            'appid': self.appid,
            'time': timestamp,
            'whatDo': what_do,
            'sign': sign
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get('dt', [])
        except Exception as e:
            print(f"获取站点列表失败: {e}")
            return []
    
    def get_article_list(
        self,
        website_ids: str,
        page: int = 1,
        keyword: str = "",
        page_size: int = 20,
        days: int = 1
    ) -> Dict:
        """获取文章列表"""
        timestamp = int(time.time())
        what_do = "SearchList"
        last_time = time.strftime('%Y-%m-%d', time.localtime(timestamp - days * 86400))
        
        sign = self._generate_sign(
            self.appid, page, website_ids, keyword, last_time, page_size, timestamp, what_do, self.password
        )
        
        params = {
            'appid': self.appid,
            'gopage': page,
            'idlist': website_ids,
            'key': keyword,
            'lastTime': last_time,
            'pagesize': page_size,
            'time': timestamp,
            'whatDo': what_do,
            'sign': sign
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"获取文章列表失败: {e}")
            return {'obj': {'rows': [], 'total': 0}}
    
    def get_article_detail(self, linkurl_id: str) -> Dict:
        """获取文章详情"""
        timestamp = int(time.time())
        what_do = "GetArticleInfo"
        sign = self._generate_sign(self.appid, linkurl_id, timestamp, what_do, self.password)
        
        params = {
            'appid': self.appid,
            'id': linkurl_id,
            'time': timestamp,
            'whatDo': what_do,
            'sign': sign
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"获取文章详情失败: {e}")
            return {}


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(
        self,
        host: str = '110.43.247.31',
        port: int = 28550,
        database: str = 'cbcms',
        user: str = '21xmt_user',
        password: str = 'CMScD!T7qCW%f1moUyw2F!AMzCAR0m'
    ):
        self.conn = pymysql.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
    
    def close(self):
        """关闭连接"""
        self.conn.close()
    
    def create_tables(self):
        """创建文章表和关联表"""
        with self.conn.cursor() as cursor:
            # 文章表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cnfanews_articles (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    articlesequenceid VARCHAR(64) UNIQUE COMMENT 'API文章ID',
                    linkurl_id VARCHAR(64) COMMENT 'linkurl提取的ID',
                    title VARCHAR(500) COMMENT '文章标题',
                    content LONGTEXT COMMENT '文章内容',
                    papername VARCHAR(200) COMMENT '来源媒体',
                    websiteid VARCHAR(20) COMMENT '站点ID',
                    source_url VARCHAR(500) COMMENT '原文链接',
                    api_typename VARCHAR(100) COMMENT 'API返回的分类',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_article_id (articlesequenceid),
                    INDEX idx_created_at (created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Cnfanews文章表'
            """)
            
            # 文章标签关联表（无外键约束，避免权限问题）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS article_label_mapping (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    article_id INT NOT NULL COMMENT '文章表ID',
                    subcat_id INT NOT NULL COMMENT 'subcat_morning表ID',
                    confidence FLOAT DEFAULT 1.0 COMMENT '置信度（模型预测用）',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY uk_article_label (article_id, subcat_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='文章标签关联表'
            """)
            
            self.conn.commit()
            print("✓ 数据表创建完成")
    
    def get_subcat_labels(self) -> List[Dict]:
        """获取 subcat_morning 表的所有标签"""
        with self.conn.cursor() as cursor:
            # 先查看表结构
            cursor.execute("DESCRIBE subcat_morning")
            columns = [col['Field'] for col in cursor.fetchall()]
            print(f"  subcat_morning 表字段: {columns}")
            
            # 构建查询（根据实际字段）
            select_fields = ['subcat_id', 'catname']
            if 'is_morning' in columns:
                where_clause = "WHERE is_morning = 1"
            else:
                where_clause = ""
            
            query = f"SELECT {', '.join(select_fields)} FROM subcat_morning {where_clause} ORDER BY subcat_id"
            cursor.execute(query)
            return cursor.fetchall()
    
    def save_article(self, article_data: Dict) -> Optional[int]:
        """
        保存文章到数据库
        
        Returns:
            文章ID，如果已存在则返回None
        """
        with self.conn.cursor() as cursor:
            # 检查是否已存在
            cursor.execute(
                "SELECT id FROM cnfanews_articles WHERE articlesequenceid = %s",
                (article_data['articlesequenceid'],)
            )
            if cursor.fetchone():
                return None
            
            # 插入新文章
            cursor.execute("""
                INSERT INTO cnfanews_articles 
                (articlesequenceid, linkurl_id, title, content, papername, websiteid, source_url, api_typename)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                article_data['articlesequenceid'],
                article_data.get('linkurl_id'),
                article_data.get('title', ''),
                article_data.get('content', ''),
                article_data.get('papername', ''),
                article_data.get('websiteid', ''),
                article_data.get('source_url', ''),
                article_data.get('api_typename', '')
            ))
            
            self.conn.commit()
            return cursor.lastrowid
    
    def link_article_to_label(self, article_db_id: int, subcat_id: int, confidence: float = 1.0):
        """建立文章与标签的关联"""
        with self.conn.cursor() as cursor:
            try:
                cursor.execute("""
                    INSERT INTO article_label_mapping (article_id, subcat_id, confidence)
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE confidence = VALUES(confidence)
                """, (article_db_id, subcat_id, confidence))
                self.conn.commit()
            except Exception as e:
                print(f"  关联标签失败: {e}")
    
    def auto_label_article(self, article_db_id: int, article_content: str, article_title: str) -> List[int]:
        """
        基于规则自动匹配标签 - 优化版（解决标签不平衡问题）
        返回匹配到的 subcat_id 列表
        """
        matched_labels = []
        text = (article_title + ' ' + article_content).lower()
        title_lower = article_title.lower()
        
        # 获取所有标签
        labels = self.get_subcat_labels()
        
        # 标签关键词映射（扩展匹配规则）
        # 优先级：稀缺标签 > 常见标签
        keyword_mapping = {
            # 稀缺标签 - 降低匹配门槛，优先匹配
            '粤港澳': ['粤港澳', '大湾区', '香港', '澳门', '深圳', '广州', '珠三角', '前海', '横琴', '南沙'],
            '学习经济': ['学习', '习近平', '讲话', '政策解读', '会议精神', '重要讲话', '指示批示'],
            '宏观': ['宏观', '经济政策', '货币政策', '财政政策', 'GDP', '通胀', 'CPI', '央行', '人民银行', '发改委'],
            
            # 常见标签 - 正常匹配门槛
            '金融': ['金融', '银行', '保险', '信贷', '利率', '汇率', '融资', '贷款', '存款', '信托'],
            '证券': ['证券', '股票', '股市', 'A股', '港股', '美股', '指数', '涨停', '跌停', '牛市', '熊市', '券商'],
            '全球': ['全球', '国际', '美国', '欧洲', '日本', '美联储', '华尔街', '海外市场', '国际贸易'],
            '评论': ['评论', '分析', '观点', '解读', '展望', '研判', '综述', '时评', '社评'],
            '投资': ['投资', '理财', '基金', '私募', '公募', '资产配置', '收益率', '回报', '风投'],
            '视频': ['视频', '直播', '访谈', '对话', '节目', '视频号', '短视频'],
            '公司': ['公司', '企业', '上市公司', '财报', '业绩', '年报', '季报', 'IPO', '并购', '重组']
        }
        
        # 稀缺标签列表（需要优先保证召回）
        scarce_labels = ['粤港澳', '学习经济', '宏观']
        
        for label in labels:
            catname = label['catname']
            if not catname:
                continue
            
            is_scarce = catname in scarce_labels
            keywords = keyword_mapping.get(catname, [])
            
            # 1. 稀缺标签：标题匹配即命中（提高召回）
            if is_scarce:
                if any(kw in title_lower for kw in keywords):
                    matched_labels.append(label['subcat_id'])
                    continue
                # 内容中匹配1个关键词即可
                if any(kw in text for kw in keywords):
                    matched_labels.append(label['subcat_id'])
                    continue
            
            # 2. 常见标签：正常匹配逻辑
            else:
                # 直接包含匹配
                if catname.lower() in text:
                    matched_labels.append(label['subcat_id'])
                    continue
                
                # 关键词扩展匹配（需2个或标题匹配）
                match_count = sum(1 for kw in keywords if kw in text)
                if match_count >= 2 or any(kw in title_lower for kw in keywords):
                    matched_labels.append(label['subcat_id'])
        
        return matched_labels
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as total FROM cnfanews_articles")
            article_count = cursor.fetchone()['total']
            
            cursor.execute("SELECT COUNT(*) as total FROM article_label_mapping")
            mapping_count = cursor.fetchone()['total']
            
            return {
                'article_count': article_count,
                'labeled_count': mapping_count
            }


def clean_html(html_content: str) -> str:
    """清理 HTML 标签"""
    if not html_content:
        return ""
    # 移除 HTML 标签
    text = re.sub(r'<[^>]+>', '', html_content)
    # 替换 HTML 实体
    text = text.replace('&nbsp;', ' ').replace('&quot;', '"').replace('&amp;', '&')
    text = text.replace('&lt;', '<').replace('&gt;', '>')
    # 清理多余空白
    text = ' '.join(text.split())
    return text.strip()


def fetch_and_store(
    max_articles: int = 2000,
    days: int = 7,
    auto_label: bool = True
):
    """
    获取文章并存储到数据库
    
    Args:
        max_articles: 最大获取数量
        days: 获取最近几天的数据
        auto_label: 是否自动打标签
    """
    print("=" * 60)
    print("Cnfanews 数据采集与存储")
    print("=" * 60)
    
    # 初始化
    api = CnfanewsAPI()
    db = DatabaseManager()
    
    try:
        # 创建表
        db.create_tables()
        
        # 获取标签信息
        labels = db.get_subcat_labels()
        print(f"\n从 subcat_morning 表获取到 {len(labels)} 个标签")
        print("标签示例:")
        for label in labels[:10]:
            print(f"  ID: {label['subcat_id']}, 名称: {label['catname']}")
        
        # 获取站点列表
        print("\n获取站点列表...")
        websites = api.get_website_list()
        if not websites:
            print("获取站点列表失败")
            return
        
        print(f"发现 {len(websites)} 个站点")
        website_ids = ','.join([str(s.get('websiteid')) for s in websites[:50]])  # 取前50个站点
        
        # 采集文章列表
        print(f"\n开始采集最近 {days} 天的文章...")
        all_articles = []
        page = 1
        
        while len(all_articles) < max_articles:
            result = api.get_article_list(
                website_ids=website_ids,
                page=page,
                page_size=20,
                days=days
            )
            
            rows = result.get('obj', {}).get('rows', [])
            if not rows:
                break
            
            all_articles.extend(rows)
            print(f"  第 {page} 页: 获取 {len(rows)} 条，总计 {len(all_articles)}")
            
            if len(rows) < 20:
                break
            
            page += 1
            time.sleep(0.5)
        
        print(f"\n共获取 {len(all_articles)} 篇文章列表")
        
        # 获取详情并存储
        print("\n获取文章详情并存储...")
        stored_count = 0
        skipped_count = 0
        
        for i, article in enumerate(tqdm(all_articles[:max_articles], desc="处理文章"), 1):
            linkurl_id = article.get('linkurl', '')[-48:] if article.get('linkurl') else None
            if not linkurl_id:
                continue
            
            try:
                # 获取详情
                detail = api.get_article_detail(linkurl_id)
                
                if not detail.get('Succeed') or not detail.get('Msg'):
                    continue
                
                # 清理内容
                content = clean_html(detail['Msg'])
                if len(content) < 100:  # 过滤太短的内容
                    continue
                
                # 准备数据
                article_data = {
                    'articlesequenceid': article.get('articlesequenceid'),
                    'linkurl_id': linkurl_id,
                    'title': article.get('title', ''),
                    'content': content,
                    'papername': article.get('papername', ''),
                    'websiteid': article.get('websiteid', ''),
                    'source_url': article.get('url', ''),
                    'api_typename': article.get('typename', '')
                }
                
                # 保存到数据库
                article_db_id = db.save_article(article_data)
                
                if article_db_id:
                    stored_count += 1
                    
                    # 自动打标签
                    if auto_label:
                        matched_labels = db.auto_label_article(
                            article_db_id, 
                            content, 
                            article.get('title', '')
                        )
                        for subcat_id in matched_labels:
                            db.link_article_to_label(article_db_id, subcat_id)
                else:
                    skipped_count += 1
                
                time.sleep(0.2)  # 避免请求过快
                
            except Exception as e:
                print(f"  处理文章失败: {e}")
                continue
        
        # 统计
        stats = db.get_stats()
        print("\n" + "=" * 60)
        print("采集完成")
        print("=" * 60)
        print(f"本次新增: {stored_count} 篇")
        print(f"跳过重复: {skipped_count} 篇")
        print(f"数据库总计: {stats['article_count']} 篇文章")
        print(f"已打标签: {stats['labeled_count']} 个关联")
        
    finally:
        db.close()


def export_training_data(output_file: str = 'data/cnfanews_train.json'):
    """从数据库导出训练数据"""
    db = DatabaseManager()
    
    try:
        with db.conn.cursor() as cursor:
            # 查询已打标签的文章
            cursor.execute("""
                SELECT 
                    a.id,
                    a.title,
                    a.content,
                    GROUP_CONCAT(sm.catname) as labels
                FROM cnfanews_articles a
                JOIN article_label_mapping alm ON a.id = alm.article_id
                JOIN subcat_morning sm ON alm.subcat_id = sm.subcat_id
                GROUP BY a.id
                LIMIT 10000
            """)
            
            results = cursor.fetchall()
            
            # 转换为训练格式
            train_data = []
            for row in results:
                train_data.append({
                    'text': row['content'],
                    'title': row['title'],
                    'labels': row['labels'].split(',') if row['labels'] else []
                })
            
            # 保存
            os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(train_data, f, ensure_ascii=False, indent=2)
            
            print(f"导出 {len(train_data)} 条训练数据到: {output_file}")
            
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description='Fetch Cnfanews data and store to MySQL')
    parser.add_argument('--max_articles', type=int, default=2000,
                        help='最大获取文章数量')
    parser.add_argument('--days', type=int, default=7,
                        help='获取最近几天的数据')
    parser.add_argument('--no_auto_label', action='store_true',
                        help='关闭自动打标签')
    parser.add_argument('--export', action='store_true',
                        help='导出训练数据')
    parser.add_argument('--output', type=str, default='data/cnfanews_train.json',
                        help='导出文件路径')
    
    args = parser.parse_args()
    
    if args.export:
        export_training_data(args.output)
    else:
        fetch_and_store(
            max_articles=args.max_articles,
            days=args.days,
            auto_label=not args.no_auto_label
        )


if __name__ == '__main__':
    main()

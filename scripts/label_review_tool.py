"""
标签人工审核工具
用于检查和修正自动标注的标签质量
"""
import pymysql
import json
from typing import List, Dict


class LabelReviewTool:
    """标签审核工具"""
    
    def __init__(self):
        self.conn = pymysql.connect(
            host='110.43.247.31', port=28550, database='cbcms',
            user='21xmt_user', password='CMScD!T7qCW%f1moUyw2F!AMzCAR0m',
            charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor
        )
    
    def close(self):
        self.conn.close()
    
    def get_label_stats(self) -> Dict:
        """获取标签统计信息"""
        with self.conn.cursor() as cursor:
            # 每篇文章的标签数分布
            cursor.execute("""
                SELECT 
                    article_id,
                    COUNT(*) as label_count
                FROM article_label_mapping
                GROUP BY article_id
            """)
            label_dist = cursor.fetchall()
            
            # 各标签的文章数
            cursor.execute("""
                SELECT 
                    sm.catname,
                    COUNT(*) as article_count
                FROM article_label_mapping alm
                JOIN subcat_morning sm ON alm.subcat_id = sm.subcat_id
                GROUP BY sm.subcat_id, sm.catname
                ORDER BY article_count DESC
            """)
            tag_counts = cursor.fetchall()
            
            return {
                'label_dist': label_dist,
                'tag_counts': tag_counts,
                'avg_labels_per_article': sum(d['label_count'] for d in label_dist) / len(label_dist) if label_dist else 0
            }
    
    def review_by_tag(self, tag_name: str, limit: int = 10) -> List[Dict]:
        """审核指定标签的文章"""
        with self.conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    a.id,
                    a.title,
                    LEFT(a.content, 500) as content_preview,
                    sm.catname as label
                FROM cnfanews_articles a
                JOIN article_label_mapping alm ON a.id = alm.article_id
                JOIN subcat_morning sm ON alm.subcat_id = sm.subcat_id
                WHERE sm.catname = %s
                ORDER BY a.created_at DESC
                LIMIT %s
            """, (tag_name, limit))
            return cursor.fetchall()
    
    def remove_label(self, article_id: int, tag_name: str):
        """删除文章的某个标签"""
        with self.conn.cursor() as cursor:
            cursor.execute("""
                DELETE alm FROM article_label_mapping alm
                JOIN subcat_morning sm ON alm.subcat_id = sm.subcat_id
                WHERE alm.article_id = %s AND sm.catname = %s
            """, (article_id, tag_name))
            self.conn.commit()
            print(f"✓ 已删除文章 {article_id} 的标签 '{tag_name}'")
    
    def add_label(self, article_id: int, tag_name: str):
        """给文章添加标签"""
        with self.conn.cursor() as cursor:
            # 获取标签ID
            cursor.execute(
                "SELECT subcat_id FROM subcat_morning WHERE catname = %s",
                (tag_name,)
            )
            result = cursor.fetchone()
            if not result:
                print(f"✗ 标签 '{tag_name}' 不存在")
                return
            
            subcat_id = result['subcat_id']
            
            # 添加关联
            try:
                cursor.execute("""
                    INSERT INTO article_label_mapping (article_id, subcat_id, confidence)
                    VALUES (%s, %s, 1.0)
                    ON DUPLICATE KEY UPDATE confidence = 1.0
                """, (article_id, subcat_id))
                self.conn.commit()
                print(f"✓ 已为文章 {article_id} 添加标签 '{tag_name}'")
            except Exception as e:
                print(f"✗ 添加失败: {e}")
    
    def find_suspicious_labels(self) -> List[Dict]:
        """找出可能有问题的标签（如单标签文章过多）"""
        with self.conn.cursor() as cursor:
            # 找出只有1个标签的文章
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
                AND sm.catname IN ('公司', '全球', '投资')  -- 常见过度匹配的标签
                LIMIT 20
            """)
            return cursor.fetchall()
    
    def interactive_review(self, tag_name: str):
        """交互式审核标签"""
        articles = self.review_by_tag(tag_name, limit=20)
        
        print(f"\n{'='*60}")
        print(f"审核标签: {tag_name} (共 {len(articles)} 篇)")
        print(f"{'='*60}\n")
        
        for i, article in enumerate(articles, 1):
            print(f"\n[{i}/{len(articles)}] ID: {article['id']}")
            print(f"标题: {article['title']}")
            print(f"内容: {article['content_preview']}...")
            print(f"当前标签: {article['label']}")
            
            action = input("\n操作: [k]保留 [d]删除标签 [q]退出 > ").strip().lower()
            
            if action == 'd':
                self.remove_label(article['id'], tag_name)
            elif action == 'q':
                break
        
        print(f"\n审核完成!")


def main():
    """主函数"""
    tool = LabelReviewTool()
    
    try:
        # 显示统计
        print("="*60)
        print("标签审核工具")
        print("="*60)
        
        stats = tool.get_label_stats()
        print(f"\n平均每篇文章标签数: {stats['avg_labels_per_article']:.2f}")
        print("\n标签分布:")
        for tag in stats['tag_counts']:
            print(f"  {tag['catname']}: {tag['article_count']} 篇")
        
        # 交互式菜单
        while True:
            print("\n" + "="*60)
            print("菜单:")
            print("  1. 审核指定标签")
            print("  2. 查看可疑标签")
            print("  3. 退出")
            
            choice = input("\n选择: ").strip()
            
            if choice == '1':
                tag = input("输入要审核的标签名: ").strip()
                tool.interactive_review(tag)
            elif choice == '2':
                suspicious = tool.find_suspicious_labels()
                print(f"\n发现 {len(suspicious)} 篇可疑单标签文章:")
                for item in suspicious[:10]:
                    print(f"  [{item['id']}] {item['title'][:50]}... -> {item['single_label']}")
            elif choice == '3':
                break
    finally:
        tool.close()


if __name__ == '__main__':
    main()

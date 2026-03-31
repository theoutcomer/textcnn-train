"""
从 Cnfanews API 获取文章数据
用于 TextCNN 训练
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import requests
import json
import hashlib
import time
import argparse
from typing import List, Dict
import urllib.parse


class CnfanewsAPI:
    """
    Cnfanews API 客户端
    """
    
    def __init__(self):
        self.appid = "21shiji_autoget"
        self.password = "Ttle57axXeH2Uwcp0sSw9w=="
        self.base_url = "http://api.cnfanews.com:8000/clientdata_rti.ashx"
    
    def _generate_sign(self, *params) -> str:
        """
        生成签名
        MD5 大写，按照 PHP 代码逻辑
        """
        sign_str = ''.join(str(p) for p in params)
        print(f"  签名原文: {sign_str[:100]}...")
        return hashlib.md5(sign_str.encode()).hexdigest().upper()
    
    def get_website_list(self) -> List[Dict]:
        """
        获取站点列表
        """
        timestamp = int(time.time())
        what_do = "GetSourceList"
        
        sign = self._generate_sign(self.appid, timestamp, what_do, self.password)
        
        params = {
            'appid': self.appid,
            'time': timestamp,
            'whatDo': what_do,
            'sign': sign
        }
        
        url = f"{self.base_url}?appid={self.appid}&time={timestamp}&whatDo={what_do}&sign={sign}"
        print(f"  请求URL: {url}")
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            print(f"  状态码: {response.status_code}")
            print(f"  响应: {response.text[:200]}...")
            response.raise_for_status()
            
            data = response.json()
            return data.get('dt', [])
        except Exception as e:
            print(f"  请求失败: {e}")
            return []
    
    def get_article_list(
        self,
        website_ids: str,
        page: int = 1,
        keyword: str = "",
        page_size: int = 20,
        days: int = 1
    ) -> Dict:
        """
        获取文章列表
        """
        timestamp = int(time.time())
        what_do = "SearchList"
        last_time = time.strftime('%Y-%m-%d', time.localtime(timestamp - days * 86400))
        
        # 按照 PHP 代码顺序: appid + gopage + idlist + key + lastTime + pagesize + time + whatDo + password
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
        
        print(f"  请求参数: {params}")
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            print(f"  状态码: {response.status_code}")
            print(f"  响应: {response.text[:300]}...")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"  请求失败: {e}")
            return {'obj': {'rows': [], 'total': 0}}
    
    def get_article_detail(self, article_id: str, linkurl_id: str = None) -> Dict:
        """
        获取文章详情
        尝试使用 linkurl_id (最后48位) 作为文章ID
        """
        # 优先使用 linkurl_id (从 linkurl 提取的最后48位)
        use_id = linkurl_id if linkurl_id else article_id
        
        timestamp = int(time.time())
        what_do = "GetArticleInfo"
        
        sign = self._generate_sign(self.appid, use_id, timestamp, what_do, self.password)
        
        params = {
            'appid': self.appid,
            'id': use_id,
            'time': timestamp,
            'whatDo': what_do,
            'sign': sign
        }
        
        url = f"{self.base_url}?appid={self.appid}&id={use_id}&time={timestamp}&whatDo={what_do}&sign={sign}"
        print(f"    详情URL: {url}")
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            print(f"    详情状态码: {response.status_code}")
            print(f"    详情响应: {response.text[:500]}...")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"    获取详情失败: {e}")
            return {}


def fetch_articles_for_training(
    website_ids: str = None,
    max_articles: int = 2000,
    output_path: str = 'data/cnfanews_train.json',
    days: int = 7
):
    """
    获取文章用于训练
    
    Args:
        website_ids: 站点ID，逗号分隔，None则获取所有站点
        max_articles: 最大获取数量
        output_path: 输出文件路径
        days: 获取最近几天的数据
    """
    api = CnfanewsAPI()
    
    print("=" * 60)
    print("Cnfanews 数据采集")
    print("=" * 60)
    
    # 获取站点列表
    if website_ids is None:
        print("\n获取站点列表...")
        websites = api.get_website_list()
        print(f"发现 {len(websites)} 个站点")
        
        # 显示站点列表
        print("\n站点列表:")
        for i, site in enumerate(websites[:20], 1):  # 显示前20个
            print(f"  {i}. ID: {site.get('websiteid')}, 名称: {site.get('websitename')}")
        
        if len(websites) > 20:
            print(f"  ... 还有 {len(websites) - 20} 个站点")
        
        # 使用所有站点ID
        website_ids = ','.join([str(s.get('websiteid')) for s in websites])
    
    print(f"\n目标站点: {website_ids[:50]}...")
    print(f"采集最近 {days} 天的数据")
    print(f"目标数量: {max_articles} 篇")
    
    # 采集文章列表
    all_articles = []
    page = 1
    
    while len(all_articles) < max_articles:
        print(f"\n获取第 {page} 页...")
        
        try:
            result = api.get_article_list(
                website_ids=website_ids,
                page=page,
                page_size=20,
                days=days
            )
            
            rows = result.get('obj', {}).get('rows', [])
            total = result.get('obj', {}).get('total', 0)
            
            if not rows:
                print("没有更多数据")
                break
            
            print(f"  本页获取 {len(rows)} 条，总计 {total} 条")
            
            # 打印第一条数据的完整结构
            if rows and page == 1:
                print(f"  第一条数据示例: {json.dumps(rows[0], ensure_ascii=False, indent=2)[:800]}")
            
            for item in rows:
                # 获取文章ID - 详情API可能需要linkurl的最后48位
                article_id = item.get('articlesequenceid', '')
                linkurl_id = item.get('linkurl', '')[-48:] if item.get('linkurl') else ''
                
                print(f"  文章: {item.get('title', '')[:30]}...")
                print(f"    articlesequenceid: {article_id}")
                print(f"    linkurl_id: {linkurl_id}")
                
                article = {
                    'articlesequenceid': article_id,
                    'title': item.get('title'),
                    'websiteid': item.get('websiteid'),
                    'websitename': item.get('websitename'),
                    'typename': item.get('typename'),
                    'createtime': item.get('createtime'),
                    'linkurl': item.get('linkurl'),
                    'raw_item': item  # 保留原始数据用于调试
                }
                all_articles.append(article)
            
            if len(rows) < 20:
                print("已获取全部数据")
                break
            
            page += 1
            
            # 简单延迟，避免请求过快
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  错误: {e}")
            break
    
    print(f"\n共获取 {len(all_articles)} 篇文章列表")
    
    # 获取文章详情（内容）
    print("\n获取文章详情...")
    articles_with_content = []
    
    for i, article in enumerate(all_articles[:max_articles], 1):
        article_id = article.get('articlesequenceid')
        linkurl_id = article.get('linkurl', '')[-48:] if article.get('linkurl') else None
        
        if not article_id:
            continue
        
        try:
            detail = api.get_article_detail(article_id, linkurl_id)
            
            # 提取内容 - API 返回的 Msg 字段包含 HTML 内容
            content = ''
            if detail.get('Succeed') and detail.get('Msg'):
                content = detail['Msg']
                # 清理 HTML 标签
                import re
                content = re.sub(r'<[^>]+>', '', content)
                content = content.replace('&nbsp;', ' ').replace('&quot;', '"')
                content = content.strip()
            
            if content:
                # 构建训练数据格式
                train_item = {
                    'text': content,
                    'title': article.get('title', ''),
                    'labels': [article.get('typename')] if article.get('typename') else [],
                    'source': article.get('websitename', ''),
                    'article_id': article_id
                }
                articles_with_content.append(train_item)
                
                if i % 10 == 0:
                    print(f"  已处理 {i}/{min(len(all_articles), max_articles)} 篇")
            
            # 延迟
            time.sleep(0.3)
            
        except Exception as e:
            print(f"  获取文章 {article_id} 失败: {e}")
            continue
    
    print(f"\n成功获取 {len(articles_with_content)} 篇带内容的文章")
    
    # 保存
    if articles_with_content:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(articles_with_content, f, ensure_ascii=False, indent=2)
        
        print(f"\n数据已保存到: {output_path}")
        
        # 统计标签
        from collections import Counter
        all_labels = []
        for item in articles_with_content:
            all_labels.extend(item.get('labels', []))
        
        label_counts = Counter(all_labels)
        print(f"\n标签分布:")
        for label, count in label_counts.most_common():
            print(f"  {label}: {count}")
    
    return articles_with_content


def main():
    parser = argparse.ArgumentParser(description='Fetch data from Cnfanews API')
    parser.add_argument('--website_ids', type=str, default=None,
                        help='站点ID，逗号分隔，默认获取所有站点')
    parser.add_argument('--max_articles', type=int, default=2000,
                        help='最大获取文章数量')
    parser.add_argument('--days', type=int, default=7,
                        help='获取最近几天的数据')
    parser.add_argument('--output', type=str, default='data/cnfanews_train.json',
                        help='输出文件路径')
    parser.add_argument('--list_sites', action='store_true',
                        help='只列出站点列表')
    
    args = parser.parse_args()
    
    if args.list_sites:
        api = CnfanewsAPI()
        websites = api.get_website_list()
        print(f"\n共有 {len(websites)} 个站点:")
        for site in websites:
            print(f"  ID: {site.get('websiteid')}, 名称: {site.get('websitename')}")
    else:
        fetch_articles_for_training(
            website_ids=args.website_ids,
            max_articles=args.max_articles,
            output_path=args.output,
            days=args.days
        )


if __name__ == '__main__':
    main()

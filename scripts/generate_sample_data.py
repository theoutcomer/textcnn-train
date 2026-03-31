"""
生成示例数据脚本
用于测试项目
"""
import json
import random
import argparse
import os


# 示例标签
LABELS = ["科技", "财经", "体育", "娱乐", "健康", "教育", "旅游"]

# 每个标签的示例文本模板
TEXT_TEMPLATES = {
    "科技": [
        "人工智能技术在医疗领域的应用越来越广泛，可以帮助医生更准确地诊断疾病。",
        "5G网络的普及将彻底改变我们的生活方式，实现万物互联。",
        "量子计算机的发展为解决复杂问题提供了新的可能性。",
        "区块链技术不仅在金融领域有应用，在供应链管理中也显示出巨大潜力。",
        "自动驾驶技术正在快速发展，预计未来几年将实现商业化应用。"
    ],
    "财经": [
        "股市今日大涨，投资者信心增强，成交量创历史新高。",
        "央行宣布降息，旨在刺激经济增长，降低企业融资成本。",
        "房地产市场调控政策持续收紧，房价涨幅明显放缓。",
        "新能源汽车产业获得大量投资，成为资本市场的新宠。",
        "人民币汇率保持稳定，外汇储备充足，国际收支平衡。"
    ],
    "体育": [
        "国家队在世界杯预选赛中取得关键胜利，出线形势大好。",
        "NBA总决赛即将开打，两支顶级球队将争夺总冠军。",
        "马拉松比赛吸引数万名跑者参与，展现了全民健身的热情。",
        "游泳世锦赛落幕，中国选手斩获多枚金牌，创造历史。",
        "网球大满贯赛事正在进行，种子选手纷纷晋级下一轮。"
    ],
    "娱乐": [
        "这部新上映的电影票房突破十亿，成为年度最大黑马。",
        "知名歌手发布新专辑，首周销量创下个人新高。",
        "综艺节目创新不断，为观众带来全新的娱乐体验。",
        "电视剧大结局引发热议，网友纷纷表示意犹未尽。",
        "电影节红毯星光熠熠，众多明星盛装出席。"
    ],
    "健康": [
        "专家建议每天保持适量运动，有助于提高身体免疫力。",
        "均衡饮食对健康至关重要，应多吃蔬菜水果。",
        "睡眠质量直接影响工作效率，成年人应保证7-8小时睡眠。",
        "定期体检可以及早发现健康问题，做到早预防早治疗。",
        "心理健康同样重要，学会调节情绪，保持积极心态。"
    ],
    "教育": [
        "在线教育平台发展迅速，为学习者提供了更多选择。",
        "素质教育越来越受到重视，培养学生的综合能力。",
        "职业教育改革深入推进，为社会培养更多技能人才。",
        "高校科研成果转化加速，推动产学研深度融合。",
        "终身学习理念深入人心，成年人积极提升自我。"
    ],
    "旅游": [
        "国庆假期旅游热度高涨，各大景区迎来客流高峰。",
        "乡村旅游成为新趋势，游客体验田园生活。",
        "出境游逐渐恢复，热门目的地机票酒店预订火爆。",
        "自驾游受到青睐，家庭出行更加灵活自由。",
        "文旅融合创新产品，为游客带来全新体验。"
    ]
}


def generate_sample(num_samples: int, output_path: str):
    """
    生成示例数据
    
    Args:
        num_samples: 样本数量
        output_path: 输出文件路径
    """
    data = []
    
    for i in range(num_samples):
        # 随机选择1-3个标签
        num_labels = random.randint(1, 3)
        selected_labels = random.sample(LABELS, num_labels)
        
        # 从选中的标签中随机选择文本模板
        primary_label = random.choice(selected_labels)
        template = random.choice(TEXT_TEMPLATES[primary_label])
        
        # 添加一些随机变化
        text = template
        
        data.append({
            "text": text,
            "labels": selected_labels
        })
    
    # 保存为 JSON
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Generated {num_samples} samples")
    print(f"Saved to: {output_path}")
    
    # 统计标签分布
    label_counts = {label: 0 for label in LABELS}
    for item in data:
        for label in item['labels']:
            label_counts[label] += 1
    
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Generate sample data for testing')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='data/sample_train.json',
                        help='Output file path')
    
    args = parser.parse_args()
    
    generate_sample(args.num_samples, args.output)


if __name__ == '__main__':
    main()

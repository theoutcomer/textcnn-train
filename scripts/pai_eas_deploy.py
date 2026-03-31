"""PAI-EAS 模型部署脚本"""
import os
import json
import subprocess


def create_eas_config(
    model_path: str,
    service_name: str = "textcnn-service",
    instance_type: str = "ecs.gn6i-c4g1.xlarge"
):
    """创建 PAI-EAS 部署配置"""
    
    config = {
        "name": service_name,
        "generate_token": "true",
        "model_config": {
            "model_path": model_path,
            "model_entry": "scripts/api_server.py",
            "model_class": "TextCNNService"
        },
        "metadata": {
            "instance": 1,
            "cpu": 4,
            "gpu": 1,
            "memory": 16000,
            "resource": instance_type
        },
        "cloud": {
            "computing": {
                "instances": [instance_type]
            }
        }
    }
    
    return config


def deploy_to_eas(config_path: str):
    """部署到 PAI-EAS"""
    cmd = f"eascmd create {config_path}"
    print(f"执行: {cmd}")
    os.system(cmd)


def update_eas_service(service_name: str, config_path: str):
    """更新 EAS 服务"""
    cmd = f"eascmd modify {service_name} -f {config_path}"
    print(f"执行: {cmd}")
    os.system(cmd)


def main():
    print("="*60)
    print("PAI-EAS 模型部署")
    print("="*60)
    
    # 检查 eascmd
    result = subprocess.run(['which', 'eascmd'], capture_output=True)
    if result.returncode != 0:
        print("✗ eascmd 未安装")
        print("请先安装 PAI-EAS 客户端:")
        print("  wget https://eas-data.oss-cn-shanghai.aliyuncs.com/tools/eascmd")
        print("  chmod +x eascmd")
        print("  ./eascmd login")
        return
    
    print("✓ eascmd 已安装")
    
    # 配置参数
    model_path = input("模型路径 (OSS/本地): ") or "/mnt/output/checkpoints/best_model.pt"
    service_name = input("服务名称 [textcnn-service]: ") or "textcnn-service"
    
    # 创建配置
    config = create_eas_config(model_path, service_name)
    config_path = f"{service_name}_config.json"
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ 配置文件已保存: {config_path}")
    
    # 部署
    if input("是否立即部署? (y/n): ").lower() == 'y':
        deploy_to_eas(config_path)


if __name__ == '__main__':
    main()

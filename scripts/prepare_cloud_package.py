"""准备云端训练打包脚本"""
import os
import sys
import shutil
import json
from datetime import datetime


def create_package_structure():
    """创建打包目录结构"""
    package_dir = "cloud_training_package"
    
    # 清理旧目录
    if os.path.exists(package_dir):
        shutil.rmtree(package_dir)
    
    # 创建目录
    dirs = [
        f"{package_dir}/src",
        f"{package_dir}/configs",
        f"{package_dir}/scripts",
        f"{package_dir}/data",
        f"{package_dir}/logs",
        f"{package_dir}/checkpoints"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    return package_dir


def copy_files(package_dir: str):
    """复制必要文件"""
    files_to_copy = [
        # 源代码
        ("src/models/textcnn.py", f"{package_dir}/src/models/"),
        ("src/data/dataset.py", f"{package_dir}/src/data/"),
        ("src/utils/__init__.py", f"{package_dir}/src/utils/"),
        ("src/utils/config.py", f"{package_dir}/src/utils/"),
        ("src/trainer.py", f"{package_dir}/src/"),
        
        # 配置
        ("configs/cloud_config.yaml", f"{package_dir}/configs/"),
        
        # 训练脚本
        ("scripts/cloud_train.py", f"{package_dir}/scripts/"),
        ("scripts/export_training_data.py", f"{package_dir}/scripts/"),
        
        # 依赖
        ("requirements.txt", f"{package_dir}/"),
    ]
    
    for src, dst in files_to_copy:
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            print(f"✓ 复制: {src} -> {dst}")
        else:
            print(f"✗ 缺失: {src}")


def create_init_files(package_dir: str):
    """创建 __init__.py 文件"""
    init_dirs = [
        f"{package_dir}/src",
        f"{package_dir}/src/models",
        f"{package_dir}/src/data",
        f"{package_dir}/src/utils",
        f"{package_dir}/src/training",
    ]
    for d in init_dirs:
        os.makedirs(d, exist_ok=True)  # 确保目录存在
        init_file = os.path.join(d, "__init__.py")
        with open(init_file, "w") as f:
            f.write("")


def create_readme(package_dir: str):
    """创建云端训练说明"""
    readme = """# TextCNN 云端训练包

## 文件结构

```
cloud_training_package/
├── src/                    # 源代码
│   ├── models/            # 模型定义
│   ├── data/              # 数据处理
│   ├── utils/             # 工具函数
│   └── training/          # 训练逻辑
├── configs/               # 配置文件
│   └── cloud_config.yaml
├── scripts/               # 训练脚本
│   └── cloud_train.py
├── data/                  # 训练数据（需上传）
├── logs/                  # 日志输出
├── checkpoints/           # 模型检查点
└── requirements.txt       # 依赖
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 上传训练数据

将训练数据放入 `data/` 目录：
- `cnfanews_training.json` - 训练数据

### 3. 启动训练

```bash
python scripts/cloud_train.py --config configs/cloud_config.yaml
```

### 4. 监控训练

- 日志: `logs/cloud/`
- 检查点: `checkpoints/cloud/`

## 高级配置

### 修改训练参数

编辑 `configs/cloud_config.yaml`:

```yaml
training:
  epochs: 50          # 训练轮数
  batch_size: 128     # 批次大小
  learning_rate: 0.001
  
gpu:
  mixed_precision: true  # 混合精度训练
```

### 多 GPU 训练

```bash
# 自动检测多 GPU
python scripts/cloud_train.py
```

### 恢复训练

```bash
python scripts/cloud_train.py --resume checkpoints/cloud/checkpoint_epoch_10.pt
```

## 输出文件

训练完成后，下载以下文件：
- `checkpoints/cloud/best_model.pt` - 最佳模型
- `logs/cloud/` - 训练日志

## 常见问题

1. **CUDA out of memory**: 减小 `batch_size`
2. **训练速度慢**: 启用 `mixed_precision: true`
3. **过拟合**: 增加 `dropout` 或减小 `embed_dim`
"""
    
    with open(f"{package_dir}/README.md", "w", encoding="utf-8") as f:
        f.write(readme)


def create_dockerfile(package_dir: str):
    """创建 Dockerfile"""
    dockerfile = """FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源代码
COPY . .

# 设置环境变量
ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=0

# 默认启动命令
CMD ["python", "scripts/cloud_train.py"]
"""
    
    with open(f"{package_dir}/Dockerfile", "w") as f:
        f.write(dockerfile)


def create_startup_script(package_dir: str):
    """创建启动脚本"""
    # Linux/Mac 启动脚本
    sh_script = """#!/bin/bash
# 云端训练启动脚本

echo "========================================"
echo "TextCNN 云端训练启动"
echo "========================================"

# 检查 GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# 启动训练
python scripts/cloud_train.py --config configs/cloud_config.yaml "$@"
"""
    
    with open(f"{package_dir}/start.sh", "w") as f:
        f.write(sh_script)
    
    # Windows 启动脚本
    bat_script = """@echo off
echo ========================================
echo TextCNN 云端训练启动
echo ========================================

python scripts/cloud_train.py --config configs/cloud_config.yaml %*
"""
    
    with open(f"{package_dir}/start.bat", "w") as f:
        f.write(bat_script)


def main():
    print("="*60)
    print("准备云端训练打包")
    print("="*60)
    
    # 创建目录
    package_dir = create_package_structure()
    print(f"\n✓ 创建目录: {package_dir}")
    
    # 复制文件
    print("\n复制文件...")
    copy_files(package_dir)
    
    # 创建 init 文件
    create_init_files(package_dir)
    print("\n✓ 创建 __init__.py 文件")
    
    # 创建说明文档
    create_readme(package_dir)
    print("✓ 创建 README.md")
    
    # 创建 Dockerfile
    create_dockerfile(package_dir)
    print("✓ 创建 Dockerfile")
    
    # 创建启动脚本
    create_startup_script(package_dir)
    print("✓ 创建启动脚本")
    
    # 创建打包信息
    info = {
        "created_at": datetime.now().isoformat(),
        "package_dir": package_dir,
        "files_count": len([f for r, d, files in os.walk(package_dir) for f in files])
    }
    
    with open(f"{package_dir}/package_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print("\n" + "="*60)
    print("打包完成!")
    print("="*60)
    print(f"目录: {package_dir}/")
    print(f"文件数: {info['files_count']}")
    print("\n下一步:")
    print("1. 将训练数据放入 cloud_training_package/data/")
    print("2. 压缩整个目录上传到云端 GPU 平台")
    print("3. 或使用 Docker 部署")


if __name__ == '__main__':
    main()

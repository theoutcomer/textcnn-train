"""PAI-DSW 上传操作指南 - 交互式脚本"""
import os
import zipfile
import subprocess


def create_upload_package():
    """创建上传包"""
    print("="*60)
    print("步骤 1: 创建上传包")
    print("="*60)
    
    package_name = "textcnn_pai_upload"
    
    # 检查数据是否存在
    data_file = "data/cnfanews_training.json"
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        print("请先运行: python scripts/export_training_data.py")
        return None
    
    # 创建临时目录
    if os.path.exists(package_name):
        import shutil
        shutil.rmtree(package_name)
    os.makedirs(package_name)
    
    # 复制必要文件
    files_to_include = [
        ("configs/pai_config.yaml", "configs/"),
        ("scripts/cloud_train.py", "scripts/"),
        ("scripts/pai_train.sh", "scripts/"),
        ("scripts/pai_setup.py", "scripts/"),
        ("src/models/textcnn.py", "src/models/"),
        ("src/data/dataset.py", "src/data/"),
        ("src/trainer.py", "src/"),
        ("src/utils/__init__.py", "src/utils/"),
        ("src/utils/config.py", "src/utils/"),
        ("requirements.txt", ""),
        (data_file, "data/"),
    ]
    
    print("\n复制文件...")
    for src, dst_dir in files_to_include:
        if os.path.exists(src):
            dst_path = os.path.join(package_name, dst_dir, os.path.basename(src))
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            # 如果是文件直接复制，如果是目录需要递归
            if os.path.isfile(src):
                import shutil
                shutil.copy2(src, dst_path)
                print(f"  ✓ {src}")
        else:
            print(f"  ✗ 缺失: {src}")
    
    # 创建 __init__.py
    init_dirs = [
        f"{package_name}/src",
        f"{package_name}/src/models",
        f"{package_name}/src/data",
        f"{package_name}/src/utils",
    ]
    for d in init_dirs:
        init_file = os.path.join(d, "__init__.py")
        with open(init_file, "w") as f:
            f.write("")
    
    # 创建启动脚本
    with open(f"{package_name}/start_train.sh", "w") as f:
        f.write("""#!/bin/bash
cd /mnt/workspace/textcnn
bash scripts/pai_train.sh
""")
    
    # 压缩
    zip_name = f"{package_name}.zip"
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(package_name):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, package_name)
                zf.write(file_path, arcname)
    
    # 清理临时目录
    import shutil
    shutil.rmtree(package_name)
    
    size_mb = os.path.getsize(zip_name) / 1024 / 1024
    print(f"\n✅ 上传包创建完成: {zip_name}")
    print(f"   大小: {size_mb:.2f} MB")
    
    return zip_name


def print_upload_steps():
    """打印上传步骤"""
    print("\n" + "="*60)
    print("步骤 2: 上传到 PAI-DSW")
    print("="*60)
    
    print("""
【方式一：通过 PAI-DSW 控制台上传】（推荐）

1. 登录阿里云 PAI 控制台
   URL: https://pai.console.aliyun.com

2. 进入 PAI-DSW 开发环境
   - 点击左侧菜单 "开发环境" -> "DSW"
   - 点击 "创建实例" 或选择已有实例

3. 上传文件
   a) 在 DSW 实例页面，点击 "打开"
   b) 进入 JupyterLab 界面
   c) 左侧文件浏览器，右键点击空白处
   d) 选择 "Upload"
   e) 选择本地的 textcnn_pai_upload.zip 文件

4. 解压文件
   - 在 Terminal 中执行：
   
   cd /mnt/workspace
   unzip textcnn_pai_upload.zip -d textcnn
   cd textcnn

【方式二：通过 OSS 中转上传】（大文件推荐）

1. 创建 OSS Bucket
   - 进入阿里云 OSS 控制台
   - 创建 Bucket（选择与 PAI 同地域）

2. 上传文件到 OSS
   - 使用 OSS 客户端或网页上传 zip 文件

3. 在 DSW 中下载
   - 安装 ossutil
   - 下载文件：
   
   ossutil cp oss://your-bucket/textcnn_pai_upload.zip .
   unzip textcnn_pai_upload.zip -d textcnn

【方式三：通过 Git 上传】（代码更新推荐）

1. 将代码推送到 Git 仓库（GitHub/Gitee/阿里云Code）

2. 在 DSW 中克隆
   
   cd /mnt/workspace
   git clone <你的仓库地址> textcnn
   cd textcnn
   
3. 单独上传数据文件（数据太大不适合 Git）
   - 使用方式一上传 data/cnfanews_training.json
""")


def print_post_upload_steps():
    """打印上传后的步骤"""
    print("\n" + "="*60)
    print("步骤 3: 上传后配置")
    print("="*60)
    
    print("""
在 PAI-DSW Terminal 中执行：

# 1. 进入项目目录
cd /mnt/workspace/textcnn

# 2. 检查文件结构
ls -la
# 应该看到: configs/  data/  requirements.txt  scripts/  src/

# 3. 检查数据文件
ls -lh data/
# 应该看到: cnfanews_training.json

# 4. 安装依赖
pip install -r requirements.txt -q

# 5. 检查环境
python scripts/pai_setup.py

# 6. 将数据复制到数据盘（推荐）
mkdir -p /mnt/data
cp data/cnfanews_training.json /mnt/data/

# 7. 启动训练
bash scripts/pai_train.sh
""")


def print_troubleshooting():
    """打印常见问题"""
    print("\n" + "="*60)
    print("常见问题")
    print("="*60)
    
    print("""
Q1: 上传失败/超时？
   - 检查文件大小（超过 500MB 建议用 OSS）
   - 分批上传：代码和数据分开上传

Q2: 解压失败？
   - 安装 unzip: apt-get install unzip
   - 或使用 Python: python -m zipfile -e textcnn_pai_upload.zip .

Q3: 找不到 GPU？
   - 确认创建实例时选择了 GPU 类型
   - 检查: nvidia-smi

Q4: 依赖安装失败？
   - 更换镜像：使用 PAI 官方 PyTorch 镜像
   - 或手动安装: pip install torch jieba fastapi

Q5: 训练中断？
   - 使用 tmux/screen 保持会话
   - 或使用 nohup: nohup bash scripts/pai_train.sh &
""")


def main():
    print("="*60)
    print("PAI-DSW 上传操作指南")
    print("="*60)
    
    # 创建上传包
    zip_file = create_upload_package()
    if not zip_file:
        return
    
    # 打印上传步骤
    print_upload_steps()
    
    # 打印上传后步骤
    print_post_upload_steps()
    
    # 打印常见问题
    print_troubleshooting()
    
    print("\n" + "="*60)
    print("准备完成！")
    print("="*60)
    print(f"\n上传文件: {zip_file}")
    print("\n下一步:")
    print("1. 登录 https://pai.console.aliyun.com")
    print("2. 创建 DSW 实例（选择 GPU）")
    print("3. 上传 zip 文件并解压")
    print("4. 运行 bash scripts/pai_train.sh")


if __name__ == '__main__':
    main()

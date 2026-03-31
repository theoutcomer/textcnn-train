"""阿里云 PAI 训练环境配置脚本"""
import os
import subprocess


def check_pai_env():
    """检查 PAI 环境"""
    print("="*60)
    print("阿里云 PAI 环境检查")
    print("="*60)
    
    # 检查 GPU
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ GPU 可用")
        print(result.stdout[:500])
    else:
        print("✗ GPU 不可用")
    
    # 检查 PyTorch
    try:
        import torch
        print(f"✓ PyTorch 版本: {torch.__version__}")
        print(f"✓ CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ GPU 数量: {torch.cuda.device_count()}")
            print(f"✓ GPU 名称: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("✗ PyTorch 未安装")
    
    # 检查其他依赖
    deps = ['jieba', 'fastapi', 'uvicorn', 'tqdm', 'loguru']
    for dep in deps:
        try:
            __import__(dep)
            print(f"✓ {dep} 已安装")
        except ImportError:
            print(f"✗ {dep} 未安装")


def install_dependencies():
    """安装依赖"""
    print("\n安装依赖...")
    os.system("pip install -r requirements.txt -q")
    print("✓ 依赖安装完成")


def setup_pai_storage():
    """设置 PAI 存储路径"""
    # PAI-DSW 默认挂载路径
    pai_paths = {
        'data': '/mnt/data/',           # 数据盘
        'code': '/mnt/workspace/',       # 代码目录
        'output': '/mnt/output/',        # 输出目录
    }
    
    print("\nPAI 存储路径:")
    for name, path in pai_paths.items():
        exists = os.path.exists(path)
        print(f"  {name}: {path} {'✓' if exists else '✗'}")
    
    return pai_paths


def create_pai_symlinks():
    """创建 PAI 目录软链接"""
    # 将项目链接到 PAI 标准路径
    if os.path.exists('/mnt/workspace/'):
        target = '/mnt/workspace/textcnn'
        if not os.path.exists(target):
            os.system(f"ln -s {os.getcwd()} {target}")
            print(f"✓ 创建链接: {target}")


if __name__ == '__main__':
    check_pai_env()
    setup_pai_storage()
    
    # 询问是否安装依赖
    if input("\n是否安装依赖? (y/n): ").lower() == 'y':
        install_dependencies()
    
    create_pai_symlinks()
    
    print("\n" + "="*60)
    print("PAI 环境配置完成!")
    print("="*60)

"""推送到 GitHub 脚本"""
import os
import subprocess


def init_git_repo():
    """初始化 Git 仓库"""
    print("="*60)
    print("初始化 Git 仓库")
    print("="*60)
    
    # 检查是否已初始化
    if os.path.exists('.git'):
        print("✓ Git 仓库已存在")
        return True
    
    # 初始化
    result = subprocess.run(['git', 'init'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ Git 仓库初始化成功")
        return True
    else:
        print(f"✗ 初始化失败: {result.stderr}")
        return False


def create_gitignore():
    """创建 .gitignore"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyTorch
*.pth
*.pt

# Data
data/*.json
data/*.csv
data/*.txt
!data/.gitkeep

# Logs
logs/
*.log

# Checkpoints
checkpoints/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Large files
*.zip
*.tar.gz
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("✓ 创建 .gitignore")


def add_and_commit():
    """添加并提交文件"""
    print("\n添加文件到 Git...")
    
    # 添加文件
    subprocess.run(['git', 'add', '.'], capture_output=True)
    
    # 提交
    result = subprocess.run(
        ['git', 'commit', '-m', 'Initial commit for PAI training'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ 文件已提交")
        return True
    else:
        print(f"提交信息: {result.stdout}")
        print(f"错误: {result.stderr}")
        return False


def setup_remote(repo_url: str):
    """设置远程仓库"""
    print("\n设置远程仓库...")
    
    # 检查是否已有 remote
    result = subprocess.run(['git', 'remote', '-v'], capture_output=True, text=True)
    if 'origin' in result.stdout:
        print("✓ 远程仓库已存在")
        return True
    
    # 添加 remote
    result = subprocess.run(
        ['git', 'remote', 'add', 'origin', repo_url],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"✓ 远程仓库已添加: {repo_url}")
        return True
    else:
        print(f"✗ 添加失败: {result.stderr}")
        return False


def push_to_github():
    """推送到 GitHub"""
    print("\n推送到 GitHub...")
    
    result = subprocess.run(
        ['git', 'push', '-u', 'origin', 'master'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ 推送成功!")
        return True
    else:
        # 尝试 main 分支
        result = subprocess.run(
            ['git', 'push', '-u', 'origin', 'main'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✓ 推送成功!")
            return True
        else:
            print(f"✗ 推送失败: {result.stderr}")
            return False


def main():
    print("="*60)
    print("推送到 GitHub")
    print("="*60)
    
    # 1. 初始化
    if not init_git_repo():
        return
    
    # 2. 创建 .gitignore
    create_gitignore()
    
    # 3. 添加并提交
    if not add_and_commit():
        return
    
    # 4. 获取仓库地址
    print("\n" + "="*60)
    print("请在 GitHub 创建仓库")
    print("="*60)
    print("1. 访问 https://github.com/new")
    print("2. 填写 Repository name: textcnn")
    print("3. 选择 Public 或 Private")
    print("4. 点击 Create repository")
    print("\n5. 复制仓库地址（HTTPS 或 SSH）")
    print("   例如: https://github.com/你的用户名/textcnn.git")
    
    repo_url = input("\n请输入仓库地址: ").strip()
    
    if not repo_url:
        print("✗ 未输入仓库地址")
        return
    
    # 5. 设置远程仓库
    if not setup_remote(repo_url):
        return
    
    # 6. 推送
    if push_to_github():
        print("\n" + "="*60)
        print("✅ 推送完成!")
        print("="*60)
        print(f"\n仓库地址: {repo_url}")
        print("\n在 PAI-DSW 中克隆:")
        print(f"  git clone {repo_url}")


if __name__ == '__main__':
    main()

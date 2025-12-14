#!/bin/bash
# TRM 项目环境设置脚本

set -e

echo "=========================================="
echo "TRM 项目环境设置"
echo "=========================================="

# 检查 conda 是否可用
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到 conda，请先安装 Anaconda 或 Miniconda"
    exit 1
fi

# 激活 conda base 环境（如果需要）
eval "$(conda shell.bash hook)"

# 创建或激活环境
ENV_NAME="trm"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "环境 ${ENV_NAME} 已存在，正在激活..."
    conda activate ${ENV_NAME}
else
    echo "创建新的 conda 环境: ${ENV_NAME}"
    conda create -n ${ENV_NAME} python=3.10 -y
    conda activate ${ENV_NAME}
fi

echo ""
echo "当前 Python 版本:"
python --version

echo ""
echo "升级 pip, wheel, setuptools..."
pip install --upgrade pip wheel setuptools

echo ""
echo "安装 PyTorch (根据你的 CUDA 版本调整)..."
echo "注意: 如果你有 NVIDIA GPU 和 CUDA 12.6，使用以下命令:"
echo "pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126"
echo ""
echo "如果没有 GPU 或使用 CPU，使用:"
echo "pip install torch torchvision torchaudio"
echo ""
read -p "按 Enter 继续安装 CPU 版本的 PyTorch，或 Ctrl+C 取消后手动安装 GPU 版本..."

pip install torch torchvision torchaudio

echo ""
echo "安装项目依赖..."
pip install -r requirements.txt

echo ""
echo "安装 adam-atan2..."
# adam-atan2 requires building CUDA extensions, so we need to install from source
ORIGINAL_DIR=$(pwd)
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"
pip download --no-deps adam-atan2
tar -xzf adam_atan2-*.tar.gz
cd adam_atan2-*
python setup.py build_ext --inplace
python setup.py install
cd "$ORIGINAL_DIR"
rm -rf "$TEMP_DIR"

echo ""
echo "=========================================="
echo "环境设置完成！"
echo "=========================================="
echo ""
echo "使用方法:"
echo "1. 激活环境: conda activate trm"
echo "2. 运行数据准备脚本或训练脚本"
echo ""
echo "可选: 登录 WandB (如果需要):"
echo "  wandb login YOUR-LOGIN"
echo ""


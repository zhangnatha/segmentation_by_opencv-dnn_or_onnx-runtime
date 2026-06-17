#!/bin/bash

# 设置退出条件：任何命令失败时退出
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

# 1. 下载cuda,并安装
echo "安装cuda..."
if [ ! -f "cuda_11.8.0_520.61.05_linux.run" ]; then
    wget -c "https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run"
else
    echo "cuda_11.8.0_520.61.05_linux.run 已存在，跳过下载"
fi
sudo sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit

# 2. 下载cudnn,并安装
echo "安装cudnn..."
if [ ! -f "cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz" ]; then
    wget -c "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz"
else
    echo "cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz 已存在，跳过下载"
fi
tar -xvJf cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz

sudo cp cudnn-linux-x86_64-8.9.7.29_cuda11-archive/include/cudnn*.h /usr/local/cuda-11.8/include/
sudo cp cudnn-linux-x86_64-8.9.7.29_cuda11-archive/lib/libcudnn* /usr/local/cuda-11.8/lib64/
sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*.h
sudo chmod a+r /usr/local/cuda-11.8/lib64/libcudnn*

# 3. 删除文件
echo "清除..."
rm -rf cudnn-linux-x86_64-8.9.7.29_cuda11-archive cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz cuda_11.8.0_520.61.05_linux.run

# 4. 配置编译环境与验证
echo "配置环境并验证..."
if ! grep -q "cuda11.8" ~/.bashrc; then
    cat <<'EOF' >> ~/.bashrc
# for cuda11.8
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda-11.8/lib64"
export PATH="${PATH}:/usr/local/cuda-11.8/bin"
export CUDA_HOME=/usr/local/cuda-11.8
EOF
fi
source ~/.bashrc
/usr/local/cuda-11.8/bin/nvcc -V

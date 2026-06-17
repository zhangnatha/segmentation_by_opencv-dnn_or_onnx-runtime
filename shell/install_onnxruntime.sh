#!/bin/bash

# 设置退出条件
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ORT_VERSION="1.17.3"
ORT_GITHUB_VERSION="v${ORT_VERSION}"
INSTALL_GPU_DIR="${ROOT_DIR}/3rdparty/onnxruntime-cuda11-1.17.3"
INSTALL_CPU_DIR="${ROOT_DIR}/3rdparty/onnxruntime"

# 定义文件路径
GPU_PKG="${SCRIPT_DIR}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz"
CPU_PKG="${SCRIPT_DIR}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"

# 下载函数（带有断点续传和大小校验）
download_if_not_exists() {
    local url=$1
    local dest=$2
    if [ ! -f "$dest" ] || [ $(stat -c%s "$dest") -lt 104857600 ]; then
        echo "正在下载: $url"
        # -c 断点续传, --no-check-certificate 忽略证书问题
        wget -c -t 5 --no-check-certificate "$url" -O "$dest"
    else
        echo "文件已存在且完整: $dest"
    fi
}

# 1. 确保文件存在 (如果不存在或文件损坏，会自动下载)
download_if_not_exists "https://github.com/microsoft/onnxruntime/releases/download/${ORT_GITHUB_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz" "$GPU_PKG"
download_if_not_exists "https://github.com/microsoft/onnxruntime/releases/download/${ORT_GITHUB_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" "$CPU_PKG"

# 2. 创建目标目录
mkdir -p "$INSTALL_GPU_DIR"
mkdir -p "$INSTALL_CPU_DIR"

# 3. 解压操作
echo "正在解压 GPU 版本..."
tar -xzf "$GPU_PKG" -C "${SCRIPT_DIR}"
mv "${SCRIPT_DIR}/onnxruntime-linux-x64-gpu-${ORT_VERSION}"/* "$INSTALL_GPU_DIR/"

echo "正在解压 CPU 版本..."
tar -xzf "$CPU_PKG" -C "${SCRIPT_DIR}"
mv "${SCRIPT_DIR}/onnxruntime-linux-x64-${ORT_VERSION}"/* "$INSTALL_CPU_DIR/"

# 4. 删除压缩包
echo "清理临时文件..."
rm -f "$GPU_PKG"
rm -f "$CPU_PKG"

# 5. 删除解压后产生的临时目录
rm -rf "${SCRIPT_DIR}/onnxruntime-linux-x64-gpu-${ORT_VERSION}"
rm -rf "${SCRIPT_DIR}/onnxruntime-linux-x64-${ORT_VERSION}"

echo "安装完成并清理完毕！"

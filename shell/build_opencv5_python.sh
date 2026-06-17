#!/bin/bash

# 设置退出条件
set -e

# 1. 动态加载 Conda 环境
if command -v conda &> /dev/null; then
    CONDA_BASE=$(conda info --base)
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
    echo "错误: 未找到 conda 命令。"
    exit 1
fi

ENV_NAME=${1:-"opencv5"}
conda activate "${ENV_NAME}"

# 2. 获取目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZIP_PATH="${SCRIPT_DIR}/opencv-5.0.0.zip"
SRC_DIR="${SCRIPT_DIR}/opencv-5.0.0"

# 3. 准备源码
if [ ! -f "$ZIP_PATH" ]; then
    echo "未找到 $ZIP_PATH，正在下载..."
    wget -O "$ZIP_PATH" https://github.com/opencv/opencv/archive/5.0.0.zip
fi

echo "正在解压..."
# 如果之前已经解压过，先删除旧目录，防止冲突
if [ -d "$SRC_DIR" ]; then rm -rf "$SRC_DIR"; fi
unzip -q "$ZIP_PATH"

# 4. 获取 Python 环境信息
PYTHON_EXECUTABLE=$(which python)
PYTHON_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_PACKAGES_PATH=$(python -c "import sysconfig; print(sysconfig.get_path('platlib'))")

# 5. 编译与安装
BUILD_DIR="${SRC_DIR}/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "开始 CMake 配置..."
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DBUILD_opencv_python3=ON \
  -DPYTHON3_EXECUTABLE="$PYTHON_EXECUTABLE" \
  -DPYTHON3_INCLUDE_DIR="$PYTHON_INCLUDE_DIR" \
  -DPYTHON3_PACKAGES_PATH="$PYTHON_PACKAGES_PATH"

echo "正在编译..."
make -j$(nproc)
make install

# 6. 清理源码目录
echo "清理源码..."
rm -rf "$SRC_DIR"
rm -rf "$ZIP_PATH"

echo "OpenCV 安装完成！"

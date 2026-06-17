#!/bin/bash

set -e

# 1. 脚本所在的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"

# OpenCV 5.0.0
OPENCV_VERSION=5.0.0

# 🎯 目标安装路径设在上一级目录（../3rdparty/opencv5）
INSTALL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/3rdparty/opencv5"

OPENCV_DIR=opencv-${OPENCV_VERSION}
CONTRIB_DIR=opencv_contrib-${OPENCV_VERSION}
BUILD_DIR=opencv_build

cd "${ROOT_DIR}"

# 下载源码
if [ ! -d "${OPENCV_DIR}" ]; then
    echo "📦 下载 OpenCV ${OPENCV_VERSION} ..."
    wget -O ${OPENCV_DIR}.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip
    unzip -q ${OPENCV_DIR}.zip
fi

if [ ! -d "${CONTRIB_DIR}" ]; then
    echo "📦 下载 opencv_contrib ${OPENCV_VERSION} ..."
    wget -O ${CONTRIB_DIR}.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip
    unzip -q ${CONTRIB_DIR}.zip
fi

# 准备全新的构建目录
rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

echo "⚙️ 配置 CMake 编译选项 (纯 C++ 库编译)..."
cmake ../${OPENCV_DIR} \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
  -DOPENCV_EXTRA_MODULES_PATH=../${CONTRIB_DIR}/modules \
  -DOPENCV_ENABLE_NONFREE=ON \
  -DCMAKE_CXX_STANDARD=17 \
  -DWITH_OPENMP=ON \
  -DWITH_TBB=ON \
  -DWITH_EIGEN=ON \
  -DWITH_GTK=ON \
  -DWITH_V4L=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_opencv_world=OFF \
  -DBUILD_TIFF=ON \
  -DBUILD_JPEG=ON \
  -DBUILD_PNG=ON \
  -DBUILD_opencv_python3=OFF \
  -DBUILD_opencv_python2=OFF \
  -DBUILD_opencv_python_bindings_generator=OFF \
  -DBUILD_opencv_python_tests=OFF

echo "🔨 编译纯 C++ 版本 OpenCV 5 + contrib..."
make -j$(nproc)

echo "📥 安装到上一级目录: ${INSTALL_DIR}"
make install

# 🧹 清理 shell 目录下的临时文件
cd "${ROOT_DIR}"
rm -rf ${CONTRIB_DIR}.zip ${OPENCV_DIR}.zip
rm -rf ${CONTRIB_DIR} ${OPENCV_DIR}
rm -rf ${BUILD_DIR}

echo "✅ 纯 C++ 版本 OpenCV ${OPENCV_VERSION} 编译并成功安装！"
echo "👉 C++ 库头文件与动态库存储在: ${INSTALL_DIR}"

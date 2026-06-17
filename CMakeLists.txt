cmake_minimum_required(VERSION 3.14)

# =========================
# 工程基本配置
# =========================
project(OpenCV5_ONNX_Inference CXX)

# 使用 C++17（OpenCV 5 / ONNX Runtime 推荐）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 默认 Release，避免性能问题
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# =========================
# OpenCV 配置（跨平台）
# =========================
# 优先使用项目自带 thirdparty OpenCV
set(OpenCV_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv5" CACHE PATH "OpenCV root path")

if(EXISTS "${OpenCV_ROOT}")
    list(PREPEND CMAKE_PREFIX_PATH "${OpenCV_ROOT}")
endif()

find_package(OpenCV REQUIRED)

message(STATUS "========== OpenCV ==========")
message(STATUS "Version : ${OpenCV_VERSION}")
message(STATUS "Include : ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Libs    : ${OpenCV_LIBS}")

# =========================
# OpenCV 推理 demo
# =========================
add_executable(opencv_inference_cpu main_opencv.cpp)

target_include_directories(opencv_inference_cpu PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(opencv_inference_cpu PRIVATE
        ${OpenCV_LIBS}
)

# Apple/Linux/Windows 统一：避免 RPATH 问题
if(UNIX AND NOT APPLE)
    target_link_options(opencv_inference_cpu PRIVATE "-Wl,--disable-new-dtags")
elseif(APPLE)
    # macOS 使用 @rpath
    set_target_properties(opencv_inference_cpu PROPERTIES
            INSTALL_RPATH "@executable_path"
    )
elseif(WIN32)
    # Windows DLL 依赖通过 PATH 或 exe 同目录
    set_target_properties(opencv_inference_cpu PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# =========================
# ONNX Runtime 配置
# =========================

# CPU 版本（默认）
set(ONNXRUNTIME_CPU_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/onnxruntime"
        CACHE PATH "ONNX Runtime CPU root"
)

# CUDA 版本（可选）
set(ONNXRUNTIME_CUDA_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/onnxruntime-cuda11-1.17.3"
        CACHE PATH "ONNX Runtime CUDA root"
)

# -------------------------
# 通用 ONNX Runtime 目标封装
# -------------------------
function(add_ort_target target_name ort_dir)
    # 1. 头文件检查
    if(NOT EXISTS "${ort_dir}/include/onnxruntime_cxx_api.h")
        message(FATAL_ERROR "ONNX Runtime include not found in: ${ort_dir}/include")
    endif()

    # 2. 跨平台查找库文件
    # Windows: onnxruntime.dll / .lib
    # Linux: libonnxruntime.so
    # macOS: libonnxruntime.dylib
    find_library(ORT_LIB
            NAMES onnxruntime
            PATHS "${ort_dir}/lib" "${ort_dir}/lib64"
            REQUIRED
    )

    message(STATUS "ONNX Runtime (${target_name}) lib: ${ORT_LIB}")

    # 3. 创建可执行程序
    add_executable(${target_name} main_ort.cpp)

    target_include_directories(${target_name} PRIVATE
            ${OpenCV_INCLUDE_DIRS}
            "${ort_dir}/include"
    )

    target_link_libraries(${target_name} PRIVATE
            ${OpenCV_LIBS}
            ${ORT_LIB}
    )

    # 4. 跨平台运行库路径处理
    if(UNIX AND NOT APPLE)
        target_link_options(${target_name} PRIVATE "-Wl,--disable-new-dtags")
        set_target_properties(${target_name} PROPERTIES
                BUILD_RPATH "${ort_dir}/lib"
                INSTALL_RPATH "${ort_dir}/lib"
        )

    elseif(APPLE)
        # macOS 使用 rpath
        set_target_properties(${target_name} PROPERTIES
                INSTALL_RPATH "@executable_path/../lib"
        )

    elseif(WIN32)
        # Windows：输出到 bin，方便 DLL 放一起
        set_target_properties(${target_name} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        )
    endif()

endfunction()

# =========================
# CPU 版本 ONNX Runtime
# =========================
if(EXISTS "${ONNXRUNTIME_CPU_DIR}/include/onnxruntime_cxx_api.h")
    add_ort_target(onnxruntime_inference_cpu "${ONNXRUNTIME_CPU_DIR}")
else()
    message(WARNING "CPU ONNX Runtime not found, skip build.")
endif()

# =========================
# CUDA 版本 ONNX Runtime（可选）
# =========================
if(EXISTS "${ONNXRUNTIME_CUDA_DIR}/include/onnxruntime_cxx_api.h")
    add_ort_target(onnxruntime_inference_cuda "${ONNXRUNTIME_CUDA_DIR}")
else()
    message(WARNING "CUDA ONNX Runtime not found, skip build.")
endif()
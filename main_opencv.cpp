#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>

struct TextObject {
    std::string label;
    cv::Point loc;
    double scale;
};

bool hasDisplay() {
    const char* showWindow = std::getenv("OPENCV_ONNX_SHOW");
    return showWindow != nullptr && std::string(showWindow) == "1";
}

void HSVtoRGB(int* r, int* g, int* b, int h, int s, int v) {
    float RGB_min, RGB_max;
    RGB_max = v * 2.55f;
    RGB_min = RGB_max * (100 - s) / 100.0f;

    int i = h / 60;
    int difs = h % 60;
    float RGB_Adj = (RGB_max - RGB_min) * difs / 60.0f;

    switch (i) {
    case 0:
        *r = static_cast<int>(RGB_max);
        *g = static_cast<int>(RGB_min + RGB_Adj);
        *b = static_cast<int>(RGB_min);
        break;
    case 1:
        *r = static_cast<int>(RGB_max - RGB_Adj);
        *g = static_cast<int>(RGB_max);
        *b = static_cast<int>(RGB_min);
        break;
    case 2:
        *r = static_cast<int>(RGB_min);
        *g = static_cast<int>(RGB_max);
        *b = static_cast<int>(RGB_min + RGB_Adj);
        break;
    case 3:
        *r = static_cast<int>(RGB_min);
        *g = static_cast<int>(RGB_max - RGB_Adj);
        *b = static_cast<int>(RGB_max);
        break;
    case 4:
        *r = static_cast<int>(RGB_min + RGB_Adj);
        *g = static_cast<int>(RGB_min);
        *b = static_cast<int>(RGB_max);
        break;
    default:
        *r = static_cast<int>(RGB_max);
        *g = static_cast<int>(RGB_min);
        *b = static_cast<int>(RGB_max - RGB_Adj);
        break;
    }
}

std::vector<cv::Scalar> paletteBgr(int sizeOfResult) {
    std::vector<cv::Scalar> colors;
    int size = std::max(1, sizeOfResult);
    colors.reserve(size);
    for (int i = 0; i < size; ++i) {
        int r = 0, g = 0, b = 0;
        int h = static_cast<int>(360.0 / size * i);
        HSVtoRGB(&r, &g, &b, h, 100, 100);
        colors.emplace_back(b, g, r);
    }
    return colors;
}

std::string executableDir() {
    char buffer[PATH_MAX] = {0};
    ssize_t length = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    std::string path = length > 0 ? std::string(buffer, static_cast<size_t>(length)) : std::string(".");
    size_t lastSlash = path.find_last_of("/\\");
    return lastSlash == std::string::npos ? "." : path.substr(0, lastSlash);
}

std::string resultPath(const std::string& fileName) {
    std::string outputDir = executableDir();
    mkdir(outputDir.c_str(), 0755);
    return outputDir + "/" + fileName;
}

// 根据 .onnx 的绝对路径，查找同级目录下的 classes.names
std::vector<std::string> loadClassNames(const std::string& modelPath) {
    std::vector<std::string> classNames;
    
    size_t lastSlash = modelPath.find_last_of("/\\");
    std::string baseDir = (lastSlash == std::string::npos) ? "" : modelPath.substr(0, lastSlash + 1);
    std::string namesFilePath = baseDir + "classes.names";

    std::ifstream ifs(namesFilePath);
    if (ifs.is_open()) {
        std::string line;
        while (std::getline(ifs, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (!line.empty()) {
                classNames.push_back(line);
            }
        }
        ifs.close();
    } else {
        std::cout << "WARN: '" << namesFilePath << "' not found. Using default numeric IDs." << std::endl;
    }
    return classNames;
}

int main(int argc, char** argv) {
    // 🎯 1. 命令行参数校验
    if (argc < 6) {
        std::cerr << "ERROR: Insufficient arguments!" << std::endl;
        std::cerr << "USAGE: " << argv[0] << " <model_path.onnx> <image_path.png/.jpg> <true/false> <conf_threshold> <iou_threshold>" << std::endl;
        return -1;
    }

    std::string modelPath = argv[1];
    std::string imagePath = argv[2];
    std::string showBoxesStr = argv[3];
    
    std::transform(showBoxesStr.begin(), showBoxesStr.end(), showBoxesStr.begin(), ::tolower);
    bool showBoxes = (showBoxesStr == "true" || showBoxesStr == "1" || showBoxesStr == "yes");

    // 🎯 动态接收并解析外部传入的阈值参数
    float confThreshold = std::stof(argv[4]);
    float iouThreshold = std::stof(argv[5]); 

    // 动态提取类别名
    std::vector<std::string> classNames = loadClassNames(modelPath);
    int numClasses = static_cast<int>(classNames.size());

    std::cout << "INFO: Starting OpenCV 5 C++ inference..." << std::endl;
    std::cout << " └─ Model: " << modelPath << std::endl;
    std::cout << " └─ Image: " << imagePath << std::endl;
    std::cout << " └─ Show Bounding Boxes: " << (showBoxes ? "True" : "False") << std::endl;
    std::cout << " └─ Confidence Threshold: " << confThreshold << std::endl;
    std::cout << " └─ IoU NMS Threshold: " << iouThreshold << std::endl;

    // 2. 读取图像
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "ERROR: Could not read image: " << imagePath << std::endl;
        return -1;
    }
    int originalW = image.cols;
    int originalH = image.rows;

    // 启动全流程计时
    auto total_start = std::chrono::high_resolution_clock::now();

    cv::Mat maskOverlay = cv::Mat::zeros(image.size(), image.type());

    // 3. 图像预处理
    cv::Mat blob;
    cv::Size inputSize(640, 640);
    cv::dnn::blobFromImage(image, blob, 1.0 / 255.0, inputSize, cv::Scalar(), true, false);

    // 4. 加载模型并前向传播
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
    if (net.empty()) {
        std::cerr << "ERROR: Could not load ONNX model!" << std::endl;
        return -1;
    }
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    net.setInput(blob);
    std::vector<cv::Mat> outputs;

    // 单独测量前向传播（推理）耗时
    auto forward_start = std::chrono::high_resolution_clock::now();
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    auto forward_end = std::chrono::high_resolution_clock::now();

    // 动态分流输出层
    cv::Mat preds, proto;
    if (outputs[0].size[1] == 32) {
        proto = outputs[0];
        preds = outputs[1];
    } else {
        preds = outputs[0];
        proto = outputs[1];
    }

    if (preds.dims == 3) {
        if (preds.size[1] < preds.size[2]) {
            preds = preds.reshape(1, preds.size[1]);
            cv::transpose(preds, preds);
        } else {
            preds = preds.reshape(1, preds.size[1]);
        }
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;
    std::vector<cv::Mat> maskCoefficients;

    float xFactor = originalW / 640.0f;
    float yFactor = originalH / 640.0f;

    // 5. 过滤目标
    for (int i = 0; i < preds.rows; ++i) {
        float* data = preds.ptr<float>(i);
        float* classesScores = data + 4;

        cv::Mat scoresMat(1, numClasses, CV_32FC1, classesScores);
        cv::Point classIdPoint;
        double maxClassScore;
        cv::minMaxLoc(scoresMat, 0, &maxClassScore, 0, &classIdPoint);

        if (maxClassScore > confThreshold) {
            float cx = data[0];
            float cy = data[1];
            float w = data[2];
            float h = data[3];

            int left = static_cast<int>((cx - w / 2.0f) * xFactor);
            int top = static_cast<int>((cy - h / 2.0f) * yFactor);
            int width = static_cast<int>(w * xFactor);
            int height = static_cast<int>(h * yFactor);

            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(static_cast<float>(maxClassScore));
            classIds.push_back(classIdPoint.x);

            cv::Mat coeffs(1, 32, CV_32FC1, data + 4 + numClasses);
            maskCoefficients.push_back(coeffs.clone());
        }
    }

    std::vector<int> indices;
    // 使用外部参数传入的映射阈值
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, iouThreshold, indices);

    std::vector<cv::Scalar> colors = paletteBgr(numClasses);

    std::vector<TextObject> textQueue;

    // 6. 恢复掩码与计算绘制坐标
    if (!indices.empty()) {
        int protoChannels = proto.size[1];
        int protoH = proto.size[2];
        int protoW = proto.size[3];
        cv::Mat protoFlat = proto.reshape(1, protoChannels);

        for (int idx : indices) {
            cv::Rect box = boxes[idx];
            int classId = classIds[idx];
            cv::Scalar color = colors[classId % colors.size()];

            cv::Mat coeffs = maskCoefficients[idx];
            cv::Mat singleMaskMat = coeffs * protoFlat;
            singleMaskMat = singleMaskMat.reshape(1, protoH);

            for (int r = 0; r < protoH; ++r) {
                float* ptr = singleMaskMat.ptr<float>(r);
                for (int c = 0; c < protoW; ++c) {
                    ptr[c] = 1.0f / (1.0f + std::exp(-ptr[c]));
                }
            }

            cv::Mat scaleMask;
            cv::resize(singleMaskMat, scaleMask, cv::Size(640, 640));
            
            int mx1 = std::max(0, static_cast<int>(box.x / xFactor));
            int my1 = std::max(0, static_cast<int>(box.y / yFactor));
            int mx2 = std::min(640, static_cast<int>((box.x + box.width) / xFactor));
            int my2 = std::min(640, static_cast<int>((box.y + box.height) / yFactor));

            cv::Mat cropMask = cv::Mat::zeros(640, 640, CV_32FC1);
            if (mx2 > mx1 && my2 > my1) {
                scaleMask(cv::Rect(mx1, my1, mx2 - mx1, my2 - my1)).copyTo(cropMask(cv::Rect(mx1, my1, mx2 - mx1, my2 - my1)));
            }

            cv::Mat actualMask;
            cv::resize(cropMask, actualMask, image.size());
            cv::Mat binaryMask = actualMask > 0.5f;

            cv::Mat coloredMask(image.size(), image.type(), color);
            coloredMask.copyTo(maskOverlay, binaryMask);

            if (showBoxes) {
                cv::rectangle(maskOverlay, box, color, 2);
            }

            std::string nameLabel = (classId < classNames.size()) ? classNames[classId] : std::to_string(classId);
            std::string label = nameLabel + ": " + cv::format("%.2f", confidences[idx]);
            
            double fontScale = 0.55;
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fontScale, 1, &baseLine);

            int textX = 0, textY = 0;
            if (showBoxes) {
                textX = box.x;
                textY = box.y - 8;
                if (textY < labelSize.height + 5) {
                    textY = box.y + box.height - 8;
                    if (textY > originalH - 5) {
                        textY = box.y + labelSize.height + 5;
                    }
                }
            } else {
                // 完全平台解耦的外接框定位
                textX = box.x + (box.width / 2) - (labelSize.width / 2);
                textY = box.y + (box.height / 2) + (labelSize.height / 2);
            }

            textX = std::max(5, std::min(textX, originalW - labelSize.width - 5));
            textY = std::max(labelSize.height + 5, std::min(textY, originalH - baseLine - 5));

            textQueue.push_back({label, cv::Point(textX, textY), fontScale});
        }

        // 7. 混合半透明掩码层
        float alpha = 0.4f;
        cv::addWeighted(maskOverlay, alpha, image, 1.0f, 0.0, image);

        // 8. 全透明悬浮黑边描边字体绘制
        for (const auto& textObj : textQueue) {
            cv::putText(image, textObj.label, textObj.loc, cv::FONT_HERSHEY_SIMPLEX, textObj.scale, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
            cv::putText(image, textObj.label, textObj.loc, cv::FONT_HERSHEY_SIMPLEX, textObj.scale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }

        auto total_end   = std::chrono::high_resolution_clock::now();

        double forwardCost = std::chrono::duration<double, std::milli>(forward_end - forward_start).count();
        double totalCost = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        std::cout << "-------------------------------------------" << std::endl;
        std::cout << "PERF: Pure inference (forward) cost: " << cv::format("%.2f", forwardCost) << " ms" << std::endl;
        std::cout << "PERF: Total algorithm pipeline cost: " << cv::format("%.2f", totalCost) << " ms (including pre/post-processing)" << std::endl;
        std::cout << "INFO: Inference finished. Successfully detected " << indices.size() << " targets." << std::endl;
        std::cout << "-------------------------------------------" << std::endl;
    } else {
        std::cout << "WARN: No targets detected." << std::endl;
    }

    std::string outName = resultPath("result_opencv_dnn_cpp_cpu.png");
    cv::imwrite(outName, image);
    std::cout << "INFO: Result image saved to: " << outName << std::endl;
    if (hasDisplay()) {
        try {
            cv::imshow("OpenCV 5 C++ YOLO11-Seg", image);
            cv::waitKey(0);
        } catch (const cv::Exception& e) {
            std::cout << "WARN: Could not open display window: " << e.what() << std::endl;
        }
    }

    return 0;
}

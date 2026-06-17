#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <memory>
#include <numeric>
#include <string>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

struct TextObject {
    std::string label;
    cv::Point loc;
    double scale;
};

std::vector<std::string> loadClassNames(const std::string& modelPath) {
    std::vector<std::string> classNames;
    size_t lastSlash = modelPath.find_last_of("/\\");
    std::string baseDir = (lastSlash == std::string::npos) ? "" : modelPath.substr(0, lastSlash + 1);
    std::ifstream ifs(baseDir + "classes.names");
    if (!ifs.is_open()) {
        std::cout << "WARN: '" << baseDir << "classes.names' not found. Using default numeric IDs." << std::endl;
        return classNames;
    }

    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            classNames.push_back(line);
        }
    }
    return classNames;
}

std::vector<int64_t> getTensorShape(const Ort::Value& tensor) {
    return tensor.GetTensorTypeAndShapeInfo().GetShape();
}

constexpr int kDefaultCudaBenchmarkIterations = 20;
constexpr int kDefaultCudaWarmupIterations = 5;
constexpr int kDefaultTimingWarmupIterations = 1;

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

void appendCudaProvider(Ort::SessionOptions& sessionOptions) {
    OrtCUDAProviderOptionsV2* cudaOptions = nullptr;
    const OrtApi& api = Ort::GetApi();
    Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cudaOptions));
    Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_CUDA_V2(sessionOptions, cudaOptions));
    api.ReleaseCUDAProviderOptions(cudaOptions);
}

bool hasDisplay() {
    const char* showWindow = std::getenv("OPENCV_ONNX_SHOW");
    return showWindow != nullptr && std::string(showWindow) == "1";
}

std::vector<cv::Scalar> paletteBgr(int sizeOfResult) {
    std::vector<cv::Scalar> colors;
    int size = std::max(1, sizeOfResult);
    colors.reserve(size);
    for (int i = 0; i < size; ++i) {
        int r = 0;
        int g = 0;
        int b = 0;
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

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "ERROR: Insufficient arguments!" << std::endl;
        std::cerr << "USAGE: " << argv[0]
                  << " <model_path.onnx> <image_path.png/.jpg> <true/false> <conf_threshold> <iou_threshold> [cpu|cuda]"
                  << std::endl;
        return -1;
    }

    std::string modelPath = argv[1];
    std::string imagePath = argv[2];
    std::string showBoxesStr = argv[3];
    std::transform(showBoxesStr.begin(), showBoxesStr.end(), showBoxesStr.begin(), ::tolower);
    bool showBoxes = (showBoxesStr == "true" || showBoxesStr == "1" || showBoxesStr == "yes");
    float confThreshold = std::stof(argv[4]);
    float iouThreshold = std::stof(argv[5]);
    std::string provider = argc >= 7 ? argv[6] : "cpu";
    std::transform(provider.begin(), provider.end(), provider.begin(), ::tolower);
    const bool requestedCuda = (provider == "cuda" || provider == "gpu");
    const int benchmarkIterations = requestedCuda ? kDefaultCudaBenchmarkIterations : 1;
    const int warmupIterations = requestedCuda ? kDefaultCudaWarmupIterations : 0;

    std::vector<std::string> classNames = loadClassNames(modelPath);
    int numClasses = classNames.empty() ? 4 : static_cast<int>(classNames.size());

    std::cout << "INFO: Starting ONNX Runtime C++ inference..." << std::endl;
    std::cout << " |- Model: " << modelPath << std::endl;
    std::cout << " |- Image: " << imagePath << std::endl;
    std::cout << " |- Provider: " << provider << std::endl;
    std::cout << " |- Show Bounding Boxes: " << (showBoxes ? "True" : "False") << std::endl;
    std::cout << " |- Confidence Threshold: " << confThreshold << std::endl;
    std::cout << " |- IoU NMS Threshold: " << iouThreshold << std::endl;

    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "ERROR: Could not read image: " << imagePath << std::endl;
        return -1;
    }
    cv::Mat originalImage = image.clone();
    int originalW = image.cols;
    int originalH = image.rows;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "opencv_onnxruntime");
    bool usingCuda = requestedCuda;
    auto makeSessionOptions = [](bool enableCuda) {
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.SetIntraOpNumThreads(1);
        if (enableCuda) {
            appendCudaProvider(options);
        }
        return options;
    };

    std::unique_ptr<Ort::Session> session;
    try {
        Ort::SessionOptions sessionOptions = makeSessionOptions(requestedCuda);
        session = std::make_unique<Ort::Session>(env, modelPath.c_str(), sessionOptions);
    } catch (const Ort::Exception& e) {
        if (!requestedCuda) {
            std::cerr << "ERROR: Failed to create ONNX Runtime CPU session: " << e.what() << std::endl;
            return -1;
        }
        std::cerr << "ERROR: Failed to create ONNX Runtime CUDA session: " << e.what() << std::endl;
        return -1;
    }
    usingCuda = requestedCuda;
    std::cout << "INFO: Active Provider: " << (usingCuda ? "cuda" : "cpu") << std::endl;

    Ort::AllocatorWithDefaultOptions allocator;
    auto inputNameAllocated = session->GetInputNameAllocated(0, allocator);
    std::string inputName = inputNameAllocated.get();
    size_t outputCount = session->GetOutputCount();
    std::vector<Ort::AllocatedStringPtr> outputNameAllocated;
    std::vector<const char*> outputNames;
    outputNameAllocated.reserve(outputCount);
    outputNames.reserve(outputCount);
    for (size_t i = 0; i < outputCount; ++i) {
        outputNameAllocated.emplace_back(session->GetOutputNameAllocated(i, allocator));
        outputNames.push_back(outputNameAllocated.back().get());
    }

    std::vector<int64_t> inputShape = {1, 3, 640, 640};
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    const char* inputNames[] = {inputName.c_str()};
    if (requestedCuda) {
        cv::Mat warmupBlob;
        cv::dnn::blobFromImage(originalImage, warmupBlob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
        size_t warmupTensorSize = static_cast<size_t>(warmupBlob.total());
        Ort::Value warmupTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, reinterpret_cast<float*>(warmupBlob.data), warmupTensorSize, inputShape.data(), inputShape.size());
        for (int i = 0; i < warmupIterations; ++i) {
            session->Run(Ort::RunOptions{nullptr}, inputNames, &warmupTensor, 1, outputNames.data(), outputNames.size());
        }
    }

    cv::Mat timingWarmupBlob;
    cv::dnn::blobFromImage(originalImage, timingWarmupBlob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
    size_t timingWarmupTensorSize = static_cast<size_t>(timingWarmupBlob.total());
    Ort::Value timingWarmupTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, reinterpret_cast<float*>(timingWarmupBlob.data), timingWarmupTensorSize, inputShape.data(), inputShape.size());
    for (int i = 0; i < kDefaultTimingWarmupIterations; ++i) {
        session->Run(Ort::RunOptions{nullptr}, inputNames, &timingWarmupTensor, 1, outputNames.data(), outputNames.size());
    }

    std::vector<Ort::Value> outputs;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;
    std::vector<cv::Mat> maskCoefficients;
    std::vector<int64_t> predShape;
    std::vector<int64_t> protoShape;
    double forwardCost = 0.0;
    double totalCost = 0.0;
    int detectedCount = 0;

    for (int iter = 0; iter < std::max(1, benchmarkIterations); ++iter) {
        cv::Mat imageFrame = originalImage.clone();
        cv::Mat maskOverlay = cv::Mat::zeros(imageFrame.size(), imageFrame.type());
        cv::Mat blob;
        cv::dnn::blobFromImage(imageFrame, blob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
        size_t inputTensorSize = static_cast<size_t>(blob.total());
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, reinterpret_cast<float*>(blob.data), inputTensorSize, inputShape.data(), inputShape.size());

        auto totalIterStart = std::chrono::high_resolution_clock::now();
        auto forwardStart = std::chrono::high_resolution_clock::now();
        try {
            outputs = session->Run(
                Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames.data(), outputNames.size());
        } catch (const Ort::Exception& e) {
            std::cerr << "ERROR: ONNX Runtime " << (requestedCuda ? "CUDA" : "CPU")
                      << " inference failed: " << e.what() << std::endl;
            return -1;
        }
        auto forwardEnd = std::chrono::high_resolution_clock::now();

        int predIndex = 0;
        int protoIndex = 1;
        std::vector<int64_t> shape0 = getTensorShape(outputs[0]);
        if (shape0.size() >= 2 && shape0[1] == 32) {
            protoIndex = 0;
            predIndex = 1;
        }

        const float* predData = outputs[predIndex].GetTensorData<float>();
        const float* protoData = outputs[protoIndex].GetTensorData<float>();
        predShape = getTensorShape(outputs[predIndex]);
        protoShape = getTensorShape(outputs[protoIndex]);

        int attrs = 0;
        int proposals = 0;
        bool transposed = false;
        if (predShape.size() == 3) {
            if (predShape[1] < predShape[2]) {
                attrs = static_cast<int>(predShape[1]);
                proposals = static_cast<int>(predShape[2]);
                transposed = true;
            } else {
                proposals = static_cast<int>(predShape[1]);
                attrs = static_cast<int>(predShape[2]);
            }
        } else {
            std::cerr << "ERROR: Unsupported prediction output shape." << std::endl;
            return -1;
        }

        boxes.clear();
        confidences.clear();
        classIds.clear();
        maskCoefficients.clear();
        float xFactor = originalW / 640.0f;
        float yFactor = originalH / 640.0f;
        std::vector<float> row(attrs);

        for (int i = 0; i < proposals; ++i) {
            if (transposed) {
                for (int j = 0; j < attrs; ++j) {
                    row[j] = predData[j * proposals + i];
                }
            } else {
                std::copy(predData + i * attrs, predData + (i + 1) * attrs, row.begin());
            }

            cv::Mat scoresMat(1, numClasses, CV_32FC1, row.data() + 4);
            cv::Point classIdPoint;
            double maxClassScore;
            cv::minMaxLoc(scoresMat, nullptr, &maxClassScore, nullptr, &classIdPoint);
            if (maxClassScore <= confThreshold) {
                continue;
            }

            float cx = row[0];
            float cy = row[1];
            float w = row[2];
            float h = row[3];
            boxes.emplace_back(static_cast<int>((cx - w / 2.0f) * xFactor),
                               static_cast<int>((cy - h / 2.0f) * yFactor),
                               static_cast<int>(w * xFactor),
                               static_cast<int>(h * yFactor));
            confidences.push_back(static_cast<float>(maxClassScore));
            classIds.push_back(classIdPoint.x);
            cv::Mat coeffs(1, 32, CV_32FC1, row.data() + 4 + numClasses);
            maskCoefficients.push_back(coeffs.clone());
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confThreshold, iouThreshold, indices);
        detectedCount = static_cast<int>(indices.size());

        std::vector<cv::Scalar> colors = paletteBgr(numClasses);

        std::vector<TextObject> textQueue;
        if (!indices.empty()) {
            int protoChannels = static_cast<int>(protoShape[1]);
            int protoH = static_cast<int>(protoShape[2]);
            int protoW = static_cast<int>(protoShape[3]);
            cv::Mat protoFlat(protoChannels, protoH * protoW, CV_32FC1, const_cast<float*>(protoData));

            cv::Mat maskFrame = cv::Mat::zeros(imageFrame.size(), imageFrame.type());

            for (int idx : indices) {
                cv::Rect box = boxes[idx];
                int classId = classIds[idx];
                cv::Scalar color = colors[classId % colors.size()];

                cv::Mat singleMaskMat = maskCoefficients[idx] * protoFlat;
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
                    scaleMask(cv::Rect(mx1, my1, mx2 - mx1, my2 - my1)).copyTo(
                        cropMask(cv::Rect(mx1, my1, mx2 - mx1, my2 - my1)));
                }

                cv::Mat actualMask;
                cv::resize(cropMask, actualMask, imageFrame.size());
                cv::Mat binaryMask = actualMask > 0.5f;
                maskFrame.setTo(color, binaryMask);

                if (showBoxes) {
                    cv::rectangle(maskFrame, box, color, 2);
                }

                std::string nameLabel = (classId < static_cast<int>(classNames.size())) ? classNames[classId] : std::to_string(classId);
                std::string label = nameLabel + ": " + cv::format("%.2f", confidences[idx]);
                double fontScale = 0.55;
                int baseLine;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fontScale, 1, &baseLine);
                int textX = showBoxes ? box.x : box.x + (box.width / 2) - (labelSize.width / 2);
                int textY = showBoxes ? box.y - 8 : box.y + (box.height / 2) + (labelSize.height / 2);
                if (showBoxes && textY < labelSize.height + 5) {
                    textY = box.y + box.height - 8;
                    if (textY > originalH - 5) {
                        textY = box.y + labelSize.height + 5;
                    }
                }
                textX = std::max(5, std::min(textX, originalW - labelSize.width - 5));
                textY = std::max(labelSize.height + 5, std::min(textY, originalH - baseLine - 5));
                textQueue.push_back({label, cv::Point(textX, textY), fontScale});
            }

            cv::addWeighted(maskFrame, 0.4f, imageFrame, 1.0f, 0.0, imageFrame);
            for (const auto& textObj : textQueue) {
                cv::putText(imageFrame, textObj.label, textObj.loc, cv::FONT_HERSHEY_SIMPLEX, textObj.scale, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
                cv::putText(imageFrame, textObj.label, textObj.loc, cv::FONT_HERSHEY_SIMPLEX, textObj.scale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            }

            image = imageFrame;
        }

        auto totalIterEnd = std::chrono::high_resolution_clock::now();
        forwardCost += std::chrono::duration<double, std::milli>(forwardEnd - forwardStart).count();
        totalCost += std::chrono::duration<double, std::milli>(totalIterEnd - totalIterStart).count();
    }

    forwardCost /= std::max(1, benchmarkIterations);
    totalCost /= std::max(1, benchmarkIterations);

    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "PERF: Pure inference (forward) cost: " << cv::format("%.2f", forwardCost) << " ms" << std::endl;
    std::cout << "PERF: Total algorithm pipeline cost: " << cv::format("%.2f", totalCost) << " ms (including pre/post-processing)" << std::endl;
    std::cout << "INFO: Inference finished. Successfully detected " << detectedCount << " targets." << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    std::string outName = resultPath(usingCuda ? "result_onnxruntime_cpp_cuda.png" : "result_onnxruntime_cpp_cpu.png");
    cv::imwrite(outName, image);
    std::cout << "INFO: Result image saved to: " << outName << std::endl;
    if (hasDisplay()) {
        try {
            cv::imshow("ONNX Runtime C++ YOLO11-Seg", image);
            cv::waitKey(0);
        } catch (const cv::Exception& e) {
            std::cout << "WARN: Could not open display window: " << e.what() << std::endl;
        }
    }
    return 0;
}

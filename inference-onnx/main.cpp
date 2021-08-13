#include <opencv2/dnn/dnn.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

std::vector<std::string> readLabels(std::string& labelFilepath) {
    std::vector<std::string> labels;
    std::string line;
    std::ifstream fp(labelFilepath);
    while (std::getline(fp, line)) {
        labels.push_back(line);
    }
    return labels;
}

std::vector<float> softmax(const std::vector<float> &input) {
    std::vector<float> result(input.size());
    // Calculate sum of e^
    float sum = 0;
    for (size_t i = 0; i < input.size(); i++)
        sum += exp(input[i]);

    // Softmax
    for (size_t i = 0; i < input.size(); i++)
        result[i] = exp(input[i]) / sum;
    return result;
}

int main(int argc, char* argv[]) {

    if (argc != 4) {
        std::cerr << "Usage: ./traffic_sign_classifier <model-path> <label-file> <image-path>" << std::endl;
        std::cerr << "   Eg: ./traffic_sign_classifier ../models/resnet18.onnx ../models/classes.txt ../sample_data/00003.jpg" << std::endl;
        exit(1);
    }
    
    std::string modelFilepath{argv[1]};
    std::string labelFilepath{argv[2]};
    std::string imageFilepath{argv[3]};

    int inpWidth = 128;
    int inpHeight = 128;
    std::vector<std::string> labels{readLabels(labelFilepath)};

    cv::Mat image = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);
    cv::Mat resizedImageRGB, resizedImage, preprocessedImage;
    cv::cvtColor(image, resizedImageRGB, cv::ColorConversionCodes::COLOR_BGR2RGB);
    resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);

    cv::Mat channels[3];
    cv::split(resizedImage, channels);

    cv::Scalar mean{0.4151, 0.3771, 0.4568};
    cv::Scalar std{0.2011, 0.2108, 0.1896};

    channels[0] = (channels[0] - 0.4151) / 0.2011;
    channels[1] = (channels[1] - 0.3771) / 0.2108;
    channels[2] = (channels[2] - 0.4568) / 0.1896;

    cv::merge(channels, 3, resizedImage);
    cv::dnn::blobFromImage(resizedImage, preprocessedImage);

    
    cv::dnn::Net net = cv::dnn::readNet(modelFilepath);
    net.setInput(preprocessedImage);
    cv::Mat prob = net.forward();
    std::cout << "Raw output: " << prob << std::endl;

    // Apply softmax to get prob
    cv::Mat probReshaped = prob.reshape(1, prob.total() * prob.channels());
    std::vector<float> probVec =
        probReshaped.isContinuous() ? probReshaped : probReshaped.clone();
    std::vector<float> probNormalized = softmax(probVec);
    std::cout << "Sofmax Prob: [";
    for (size_t i = 0; i < probNormalized.size(); i++) {
        std::cout << probNormalized[i];
        if (i != probNormalized.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    cv::Point classIdPoint;
    double confidence;
    minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
    int classId = classIdPoint.x;
    std::cout << "Result: classID " << classId << " - [" << labels[classId] << "] - confidence "
                << probNormalized[classId] << std::endl;

}
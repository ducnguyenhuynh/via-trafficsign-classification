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

std::vector<float> sigmoid(const std::vector<float>& m1) {
    const unsigned long vectorSize = m1.size();
    std::vector<float> output(vectorSize);
    for (unsigned i = 0; i != vectorSize; ++i) {
        output[i] = 1 / (1 + exp(-m1[i]));
    }
    return output;
}

int main(int argc, char* argv[]) {
    int inpWidth = 128;
    int inpHeight = 128;
    std::string modelFilepath{
        "/home/huynhduc/Desktop/via-trafficsign-classification/resnet/"
        "resnet18.onnx"};
    std::string labelFilepath{
        "/home/huynhduc/Desktop/via-trafficsign-classification/"
        "dataset/classes.txt"};

    std::vector<std::string> labels{readLabels(labelFilepath)};
    std::string imageFilepath{argv[1]};

    

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
    std::cout << prob << std::endl;

    // Apply sigmoid
    cv::Mat probReshaped = prob.reshape(1, prob.total() * prob.channels());
    std::vector<float> probVec =
        probReshaped.isContinuous() ? probReshaped : probReshaped.clone();
    std::vector<float> probNormalized = sigmoid(probVec);

    cv::Point classIdPoint;
    double confidence;
    minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
    int classId = classIdPoint.x;
    std::cout << " ID " << classId << " - " << labels[classId] << " confidence "
                << confidence << std::endl;

}
//
// Created by liryi on 24-11-24.
//
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// 二值化图像
Mat binarizeImage(const Mat& image, double thresholdValue) {
    Mat binaryImage;
    threshold(image, binaryImage, thresholdValue, 255, THRESH_BINARY);
    return binaryImage;
}

// 应用闭运算（填充空洞）
Mat applyMorphClose(const Mat& binaryImage, const Size& kernelSize) {
    Mat kernel = getStructuringElement(MORPH_RECT, kernelSize);
    Mat closedImage;
    morphologyEx(binaryImage, closedImage, MORPH_CLOSE, kernel);
    return closedImage;
}

// 拼接并显示结果
void displayResults(const Mat& originalImage, const Mat& processedImage) {
    Mat display;
    hconcat(originalImage, processedImage, display);

    putText(display, "Original", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255), 2);
    putText(display, "Hole Filled", Point(originalImage.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255), 2);

    namedWindow("Results", WINDOW_NORMAL);
    imshow("Results", display);
    waitKey(0);
}

int main() {
    string inputPath = "E:\\DL_class\\exam_LRX\\cpp\\images\\image10.png"; // 替换为您的图像路径
    Mat image = imread(inputPath, IMREAD_GRAYSCALE);

    if (image.empty()) {
        cerr << "Error: Unable to load image!" << endl;
        return -1;
    }

    // 二值化
    Mat binaryImage = binarizeImage(image, 128.0);

    // 闭运算
    Mat closedImage = applyMorphClose(binaryImage, Size(10, 10));

    // 显示结果
    displayResults(binaryImage, closedImage);

    return 0;
}


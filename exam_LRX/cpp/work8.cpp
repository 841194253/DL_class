//
// Created by liryi on 24-11-22.
// 8.给出图像增强处理中的空间域增强的加法、减法和乘法的实例
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// 加法增强
Mat enhanceImageAddition(const Mat& image, int constant) {
    Mat enhancedImage;
    add(image, Scalar(constant), enhancedImage);
    return enhancedImage;
}

// 减法增强
Mat enhanceImageSubtraction(const Mat& image, int constant) {
    Mat enhancedImage;
    subtract(image, Scalar(constant), enhancedImage);
    return enhancedImage;
}

// 乘法增强
Mat enhanceImageMultiplication(const Mat& image, double constant) {
    Mat enhancedImage;
    multiply(image, Scalar(constant), enhancedImage);
    return enhancedImage;
}

// 在一个窗口中显示多张图片
void displayImagesInOneWindow(const Mat& originalImage, const Mat& enhancedAddition,
                              const Mat& enhancedSubtraction, const Mat& enhancedMultiplication) {
    // 调整大小和布局
    int width = originalImage.cols;
    int height = originalImage.rows;
    Mat display = Mat::zeros(height * 2, width * 2, originalImage.type());

    // 放置原始图像
    originalImage.copyTo(display(Rect(0, 0, width, height)));
    putText(display, "Original image", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255), 2);

    // 放置加法增强
    enhancedAddition.copyTo(display(Rect(width, 0, width, height)));
    putText(display, "Addition enhancement", Point(width + 10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255), 2);

    // 放置减法增强
    enhancedSubtraction.copyTo(display(Rect(0, height, width, height)));
    putText(display, "Subtractive enhancement", Point(10, height + 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255), 2);

    // 放置乘法增强
    enhancedMultiplication.copyTo(display(Rect(width, height, width, height)));
    putText(display, "Multiplication enhancement", Point(width + 10, height + 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255), 2);

    // 显示结果
    namedWindow("Enhancement effect comparison", WINDOW_NORMAL);
    imshow("Enhancement effect comparison", display);
    waitKey(0);
}

int main() {
    // 读取图像
    string imagePath = "E:\\DL_class\\exam_LRX\\cpp\\images\\image9.png"; // 修改为你的图像路径
    Mat image = imread(imagePath, IMREAD_GRAYSCALE);

    if (image.empty()) {
        cout << "imager error" << endl;
        return -1;
    }

    // 常数值
    int constantAddSub = 50;
    double constantMul = 1.5;

    // 加法增强
    Mat enhancedAddition = enhanceImageAddition(image, constantAddSub);

    // 减法增强
    Mat enhancedSubtraction = enhanceImageSubtraction(image, constantAddSub);

    // 乘法增强
    Mat enhancedMultiplication = enhanceImageMultiplication(image, constantMul);

    // 显示所有结果
    displayImagesInOneWindow(image, enhancedAddition, enhancedSubtraction, enhancedMultiplication);

    return 0;
}


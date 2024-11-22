//
// Created by liryi on 24-11-22.
// 9.给出图像处理中空间域滤波的均值滤波、中值滤波和高斯滤波实例。
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// 均值滤波
Mat meanFilter(const Mat& image) {
    Mat result;
    blur(image, result, Size(11, 11));  // 均值滤波
    return result;
}

// 中值滤波
Mat medianFilter(const Mat& image) {
    Mat result;
    medianBlur(image, result, 15);  // 中值滤波
    return result;
}

// 高斯滤波
Mat gaussianFilter(const Mat& image) {
    Mat result;
    GaussianBlur(image, result, Size(15, 15), 0);  // 高斯滤波
    return result;
}

// 在一个窗口中显示原图和滤波后的图像
void displayFilters(const Mat& originalImage, const Mat& meanImage, const Mat& medianImage, const Mat& gaussianImage) {
    // 创建一个大的显示矩阵
    int width = originalImage.cols;
    int height = originalImage.rows;
    Mat display = Mat::zeros(height, width * 4, originalImage.type());

    // 放置原始图像
    originalImage.copyTo(display(Rect(0, 0, width, height)));
    putText(display, "original image", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255), 2);

    // 放置均值滤波图像
    meanImage.copyTo(display(Rect(width, 0, width, height)));
    putText(display, "Mean filtering", Point(width + 10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255), 2);

    // 放置中值滤波图像
    medianImage.copyTo(display(Rect(width * 2, 0, width, height)));
    putText(display, "median filtering ", Point(width * 2 + 10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255), 2);

    // 放置高斯滤波图像
    gaussianImage.copyTo(display(Rect(width * 3, 0, width, height)));
    putText(display, "Gaussian filter", Point(width * 3 + 10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255), 2);

    // 显示结果
    namedWindow("Comparison of filtering effects", WINDOW_NORMAL);
    imshow("Comparison of filtering effects", display);
    waitKey(0);
}

int main() {
    // 读取图像
    string imagePath = "E:\\DL_class\\exam_LRX\\cpp\\images\\image9.png";  // 替换为你的图像路径
    Mat image = imread(imagePath, IMREAD_GRAYSCALE);

    if (image.empty()) {
        cout << "image error" << endl;
        return -1;
    }

    // 进行滤波
    Mat meanImage = meanFilter(image);
    Mat medianImage = medianFilter(image);
    Mat gaussianImage = gaussianFilter(image);

    // 显示所有结果
    displayFilters(image, meanImage, medianImage, gaussianImage);

    return 0;
}


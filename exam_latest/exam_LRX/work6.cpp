//
// Created by liryi on 24-11-25.
//6.采用直方图均衡方法处理图像A，结果类似图B。
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 读取彩色图像
    cv::Mat image = cv::imread("..\\images\\image6.png");
    if (image.empty()) {
        std::cout << "Could not open or find the image!" << std::endl;
        return -1;
    }

    std::cout << image.type() << " " << image.channels() << std::endl;

    cv::Mat gray_image;

    // 转换为灰度图
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

    // 确保灰度图是CV_8UC1类型，如果不是，则转换为CV_8UC1
    if (gray_image.type() != CV_8UC1) {
        gray_image.convertTo(gray_image, CV_8UC1);
        std::cout << "Converted to CV_8UC1" << std::endl;
    }

    // 高斯模糊，去除噪点
    cv::Mat denoised_image;
    cv::GaussianBlur(gray_image, denoised_image, cv::Size(5, 5), 1);

    // 直方图均衡化
    cv::Mat contrast_image;
    cv::equalizeHist(denoised_image, contrast_image);

    // 锐化图像
    cv::Mat blurred_image, sharp_image;
    cv::GaussianBlur(contrast_image, blurred_image, cv::Size(5, 5), 1);
    cv::addWeighted(contrast_image, 1.5, blurred_image, -0.5, 0, sharp_image);

    // 显示处理后的图像
    cv::namedWindow("Processed Image", cv::WINDOW_NORMAL);
    cv::imshow("Processed Image", sharp_image);

    // 等待按键
    cv::waitKey(0);
    return 0;
}
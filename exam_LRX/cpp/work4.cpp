//
// Created by liryi on 24-11-7.
//
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 读取图像
    cv::Mat image = cv::imread("E:\\DL_class\\exam_LRX\\cpp\\images\\image4.png");

    if (image.empty()) {
        std::cerr << "Image not found!" << std::endl;
        return -1;
    }

    // 转换为灰度图像
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // 高斯模糊去噪
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    // Canny边缘检测
    cv::Mat edges;
    cv::Canny(blurred, edges, 50, 150);

    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 创建白色背景图像
    cv::Mat output = cv::Mat::ones(image.size(), image.type()) * 0; // 黑色背景

    // 为每个叶片分配不同的颜色并填充
    for (size_t i = 0; i < contours.size(); ++i) {
        // 随机生成一个颜色
        cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);
        // 填充每个轮廓区域
        cv::drawContours(output, contours, static_cast<int>(i), color, cv::FILLED);
    }

    // 拼接原图与分割结果图像进行对比
    cv::Mat combined;
    cv::hconcat(image, output, combined);  // 横向拼接原图与分割结果图像

    // 显示结果
    cv::imshow("Comparison: Original vs Segmentation", combined);

    cv::waitKey(0);
    return 0;
}

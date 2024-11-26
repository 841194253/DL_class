//
// Created by liryi on 24-11-24.
//4.分割获取谷子的各叶片，即把每个叶片独立分割，并用不同颜色表示。
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 读取图像
    cv::Mat image = cv::imread("..\\images\\image4.png");

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


//逻辑：
//图像读取与灰度转换： 读取输入图像并转化为灰度图像，以便后续处理。
//降噪处理： 通过高斯模糊平滑图像，减少噪声对边缘检测的干扰。
//边缘检测： 使用 Canny 算法检测图像中的边缘，突出叶片轮廓。
//轮廓检测： 使用 cv::findContours 获取每片叶片的轮廓。
//颜色填充：
//为每片叶片的轮廓随机分配颜色。
//使用 cv::drawContours 填充轮廓区域，生成分割后的叶片图像。
//结果可视化： 横向拼接原始图像与分割结果图像，直观展示分割效果。
//结果分析：
//输出图像显示了每片叶片以不同颜色标注，分割效果清晰直观。
//若某些叶片未正确分割或颜色交错，可能与边缘检测阈值或轮廓提取参数有关，可适当调整。
//拼接的对比图有助于验证分割结果是否准确。

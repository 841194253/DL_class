//
// Created by liryi on 24-11-7.
//
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 读取图像
    cv::Mat image = cv::imread("E:\\DL_class\\exam_LRX\\image\\image6.png");

    if (image.empty()) {
        std::cerr << "Image not found!" << std::endl;
        return -1;
    }

    // 创建一个掩膜图像，标记被遮挡区域为白色 (255)
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);

    // 手动绘制遮挡区域，这里假设你知道遮挡区域的位置，使用白色 (255) 填充
    // 例如，一个遮挡区域，可能是苹果的一部分：
    cv::ellipse(mask, cv::Point(200, 200), cv::Size(100, 80), 0, 0, 180, cv::Scalar(255), -1);  // 填充椭圆遮挡区域

    // 使用 inpaint 函数还原被遮挡的区域，增加修复半径 (这里设置为6，较大值)
    cv::Mat restored;
    cv::inpaint(image, mask, restored, 6, cv::INPAINT_TELEA);  // 使用更大的半径进行修复

    // 显示原图、掩膜和修复后的图像
    cv::imshow("Original Image", image);
    cv::imshow("Mask (Occlusion Area)", mask);
    cv::imshow("Restored Image", restored);

    cv::waitKey(0);
    return 0;
}
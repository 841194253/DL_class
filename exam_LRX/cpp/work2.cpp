//
// Created by liryi on 24-11-7.
//2.提取图像中的叶片病害的图斑数目、估算病害图斑占叶片的总面比例。

#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 读取图像
    cv::Mat image = cv::imread("E:\\DL_class\\exam_LRX\\cpp\\images\\image2.png");
    if (image.empty()) {
        std::cout << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // 将图像转换为 HSV 颜色空间
    cv::Mat image_hsv;
    cv::cvtColor(image, image_hsv, cv::COLOR_BGR2HSV);

    // 定义黄色区域的 HSV 范围
    cv::Scalar lower_yellow(15, 80, 80);  // 放宽范围以捕捉更多黄色区域
    cv::Scalar upper_yellow(35, 255, 255);
    cv::Mat mask_yellow;
    cv::inRange(image_hsv, lower_yellow, upper_yellow, mask_yellow);

    // 对掩膜进行形态学操作（使用较小的核）
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask_yellow, mask_yellow, cv::MORPH_CLOSE, kernel);

    // 计算叶片总面积和病变面积
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    int leaf_area = cv::countNonZero(gray);
    int disease_area = cv::countNonZero(mask_yellow);
    double disease_ratio = (static_cast<double>(disease_area) / leaf_area) * 100.0;

    // 输出结果
    std::cout << "Disease Area Ratio: " << disease_ratio << "%" << std::endl;

    cv::Mat mask_yellow_color;
    cv::cvtColor(mask_yellow, mask_yellow_color, cv::COLOR_GRAY2BGR);

    // 拼接原图和掩膜图像
    cv::Mat combined;
    cv::hconcat(image, mask_yellow_color, combined);

    // 显示拼接后的图像
    cv::imshow("Original and Disease Mask", combined);
    cv::waitKey(0);

    return 0;

}


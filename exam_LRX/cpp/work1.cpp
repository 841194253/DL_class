//
// Created by liryi on 24-11-6.
// 1.提取图中的植物部分，并估算植物的绿色部分面积，已知植物生长的槽的宽是20 cm。
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 读取图像
    std::string fileName = "E:\\DL_class\\exam_LRX\\cpp\\images\\image1.png";
    std::cout << fileName << std::endl;
    cv::Mat image = cv::imread(fileName);

    // 检查图像是否加载成功
    if (image.empty()) {
        std::cout << "image error" << std::endl;
        return -1;
    }

    // 转换为HSV颜色空间
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);

    // 定义绿色的HSV阈值范围
    cv::Scalar lower_green(35, 40, 40);
    cv::Scalar upper_green(85, 255, 255);

    // 创建绿色掩码
    cv::Mat green_mask;
    cv::inRange(hsv_image, lower_green, upper_green, green_mask);

    // 使用形态学操作去噪声
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(green_mask, green_mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(green_mask, green_mask, cv::MORPH_CLOSE, kernel);

    // 计算绿色区域的像素数
    int green_pixels = cv::countNonZero(green_mask);

    // 估算绿色部分面积
    // 计算图像的分辨率与实际宽度的比例
    double image_width_in_cm = 20.0;  // 槽的实际宽度为20厘米
    int image_width_in_pixels = image.cols;
    double pixel_to_cm_ratio = image_width_in_cm / image_width_in_pixels;

    // 将绿色像素数转换为平方厘米面积
    double green_area_cm2 = green_pixels * (pixel_to_cm_ratio * pixel_to_cm_ratio);

    std::cout << "绿色植物部分的估算面积为: " << green_area_cm2 << " 平方厘米" << std::endl;

    // 显示绿色掩码
    cv::imshow("Green Mask", green_mask);
    cv::waitKey(0);

    return 0;
}

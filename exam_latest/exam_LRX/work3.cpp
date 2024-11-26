//
// Created by liryi on 24-11-24
//3.采用图像形态学获取图像中小米的粒数、每粒小米的最大投影面积或者外接最大正方形的长与宽。
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    // 读取图像
    cv::Mat image = cv::imread("../images/image3.png");
    if (image.empty()) {
        std::cout << "无法读取图像" << std::endl;
        return -1;
    }

    // 转换为灰度图
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // 二值化处理
    cv::Mat binary;
    cv::threshold(gray, binary, 150, 255, cv::THRESH_BINARY_INV);
    // 或者使用自适应阈值
    // cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);

    // 轮廓检测
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // 计算小米颗粒的数量和面积
    int grain_count = 0;
    std::vector<double> areas;
    std::vector<int> squares;

    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > 10) {  // 忽略一些小面积噪声
            grain_count++;
            areas.push_back(area);

            // 获取最小外接矩形
            cv::Rect bounding_box = cv::boundingRect(contours[i]);
            int square_size = std::max(bounding_box.width, bounding_box.height);
            squares.push_back(square_size);

            // 在原图上绘制检测框
            cv::rectangle(image, bounding_box, cv::Scalar(0, 255, 0), 1);
        }
    }

    // 输出结果
    std::cout << "num:" << grain_count << std::endl;
    if (!areas.empty()) {
        auto max_area = *std::max_element(areas.begin(), areas.end());
        auto max_square = *std::max_element(squares.begin(), squares.end());
        std::cout << "max area: " << max_area << std::endl;
        std::cout << "max long: " << max_square << std::endl;
    } else {
        std::cout << "None" << std::endl;
    }

    // 将二值化图像转换为3通道，以便与彩色图像拼接
    cv::Mat binary_color;
    cv::cvtColor(binary, binary_color, cv::COLOR_GRAY2BGR);

    // 将原图和二值化图像拼接在一起
    cv::Mat combined;
    cv::hconcat(image, binary_color, combined);

    // 显示拼接后的图像
    cv::imshow("Combined Image (Original and Binary)", combined);
    cv::waitKey(0);
    return 0;
}

//逻辑：
//图像加载与灰度处理： 读取图像并转换为灰度图，简化后续处理。
//二值化： 通过固定阈值或自适应阈值，将图像转化为二值图，突出小米颗粒并抑制背景。
//轮廓检测： 使用 cv::findContours 检测小米颗粒的轮廓。
//颗粒分析：
//遍历轮廓，计算每颗颗粒的面积，过滤小面积噪声。
//获取外接矩形，记录最大投影面积或外接正方形的边长。
//在原图上绘制外接框，以便直观验证结果。
//输出结果：
//输出小米颗粒的总数。
//找出最大的颗粒投影面积及最大外接正方形边长。
//可视化： 拼接原图与二值化图像，显示检测结果。
//结果分析：
//输出格式：
//Copy code
//num: X   // 小米总粒数
//max area: Y   // 最大投影面积
//max long: Z   // 最大正方形边长
//拼接图中，绿色矩形框表示每粒小米的检测结果，直观展现检测精度和形态学操作的有效性。

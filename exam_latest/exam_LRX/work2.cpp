//
// Created by liryi on 24-11-24.
//2.提取图像中的叶片病害的图斑数目、估算病害图斑占叶片的总面比例。

#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 读取图像
    cv::Mat image = cv::imread("..\\images\\image2.png");
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

//逻辑：
//图像加载与检查： 从指定路径读取叶片图像，确保加载成功。
//HSV 转换： 将图像转换为 HSV 颜色空间，便于识别病变区域（黄色）。
//病害提取： 定义黄色区域的 HSV 范围，用 cv::inRange 生成病害掩码图。
//形态学处理： 使用形态学操作（闭运算）去除噪声和孔洞，清理掩码区域。
//面积计算：
//转换原图为灰度图，统计非零像素数，作为叶片总面积。
//统计掩码中非零像素数，作为病害区域面积。
//计算病害面积占比：(病害面积 / 叶片总面积) × 100%。
//结果显示：
//输出病害面积占比到控制台。
//将原图与病害掩码拼接，展示病害区域可视化结果。
//结果分析：
//输出Disease Area Ratio表示病害面积占叶片面积的比例。
//拼接图展示病害区域（黄色）与原图对比，验证病害检测的准确性。

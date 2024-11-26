//
// Created by liryi on 24-11-24.
// 1.提取图中的植物部分，并估算植物的绿色部分面积，已知植物生长的槽的宽是20 cm。
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 读取图像
    std::string fileName = "..\\images\\image1.png";
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

    std::cout << "Green area " << green_area_cm2 << std::endl;

    // 将绿色掩码转换为三通道图像（与原图相同）
    cv::Mat green_mask_rgb;
    cv::cvtColor(green_mask, green_mask_rgb, cv::COLOR_GRAY2BGR);

    // 确保原图和掩码的大小一致
    if (image.size() != green_mask_rgb.size()) {
        cv::resize(green_mask_rgb, green_mask_rgb, image.size());
    }

    // 拼接原图和绿色掩码
    cv::Mat combined_image;
    cv::hconcat(image, green_mask_rgb, combined_image);  // 水平拼接原图和绿色掩码

    // 显示拼接后的图像
    cv::imshow("Original and Green Mask", combined_image);
    cv::waitKey(0);

    return 0;
}

//逻辑：
//读取图像： 加载植物图像 image1.png，检查加载成功与否。
//颜色空间转换： 转换图像至 HSV 色彩空间，便于分离绿色区域。
//绿色提取： 定义绿色的 HSV 范围，通过 cv::inRange 生成绿色掩码。
//噪声处理： 使用形态学操作（开闭运算）去除小噪声和孔洞，得到纯净的绿色区域掩码。
//面积计算：
//通过 cv::countNonZero 统计绿色像素数。
//依据槽宽 20 cm 和图像宽度计算每像素对应的实际面积，估算绿色部分总面积（单位：平方厘米）。
//结果显示：
//将绿色掩码转为三通道，与原图水平拼接，展示对比结果。
//显示拼接图像并输出绿色面积值到控制台。
//结果分析：
//输出绿色区域的估算面积，例如 "Green area 45.73"。
//拼接图像展示了原图和绿色区域掩码的直观对比，证明绿色提取有效。

// #include <direct.h>  // Windows
// #include <iostream>
//
// int main() {
//     char cwd[1024];
//     if (_getcwd(cwd,sizeof(cwd)) != NULL) {
//         std::cout << " " << cwd << std::endl;
//     } else {
//         std::cerr << "error" << std::endl;
//     }
// }


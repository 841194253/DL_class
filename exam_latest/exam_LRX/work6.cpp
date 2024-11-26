//
// Created by liryi on 24-11-25.
//6.采用直方图均衡方法处理图像A，结果类似图B。
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 读取灰度图像
    cv::Mat image = cv::imread("..\\images\\image6.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // 对灰度图像进行直方图均衡化
    cv::Mat equalizedImage;
    cv::equalizeHist(image, equalizedImage);

    // 显示均衡化后的图像
    cv::imshow("Image", equalizedImage);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}

// 代码逻辑分析
// 加载图像
// 使用 cv::imread 加载位于路径 "..\\images\\image6.png" 的图像，并指定加载为灰度图像（cv::IMREAD_GRAYSCALE）。
// 如果加载失败（即 image.empty() 返回 true），输出错误信息并结束程序。
// 直方图均衡化：
// 使用 cv::equalizeHist 函数对加载的灰度图像进行直方图均衡化。该方法通过重新分配图像的灰度级别，使得图像的像素分布更均匀，从而增强图像的对比度。
// 显示均衡化结果：
// 使用 cv::imshow 显示均衡化后的图像。
// cv::waitKey(0) 等待用户按键，这样程序不会立即关闭，图像窗口会保持打开直到用户按下任意键。
// 使用 cv::destroyAllWindows 销毁所有图像窗口。
// 代码功能：
// 读取并加载一幅灰度图像，应用直方图均衡化处理，最终显示增强对比度的结果图像。
// 结果分析：
// 直方图均衡化 主要作用是调整图像的亮度和对比度，尤其适用于低对比度或灰度分布不均匀的图像。通过均衡化，图像的对比度增强，细节更加突出，尤其是图像的暗部和亮部区域变得更加清晰可见。
//
// Created by liryi on 24-11-12.
//6.采用直方图均衡方法处理图像A，结果类似图B。
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 读取灰度图像
    cv::Mat image = cv::imread("E:\\DL_class\\exam_LRX\\cpp\\images\\image8.png", cv::IMREAD_GRAYSCALE);
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

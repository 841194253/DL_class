//
// Created by liryi on 24-11-24.
// 13.给出采用形态学处理应用提取图像边界的实例一个。
//
// Created by liryi on 24-11-24.
//
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // 读取图像
    Mat src = imread("E:\\DL_class\\exam_LRX\\cpp\\images\\image2.png");
    if (src.empty()) {
        cerr << "无法读取图像文件！" << endl;
        return -1;
    }

    // 定义结构元素
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));

    // 执行腐蚀操作
    Mat eroded;
    erode(src, eroded, element);

    // 计算原始图像与腐蚀图像的差异
    Mat diff;
    absdiff(src, eroded, diff);

    // 拼接图像（原图、腐蚀图、差异图）
    Mat display;
    hconcat(src, eroded, display);    // 拼接原图和腐蚀图
    hconcat(display, diff, display);  // 再拼接差异图

    // 添加标签
    putText(display, "Original", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255), 2);
    putText(display, "Eroded", Point(src.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255), 2);
    putText(display, "Difference", Point(src.cols + eroded.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255), 2);

    // 显示拼接后的图像
    imshow("Results", display);

    // 等待按键事件
    waitKey(0);
    return 0;
}


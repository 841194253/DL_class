//
// Created by liryi on 24-11-24.
//
#include "opencv2/opencv.hpp" // 引入OpenCV相关库
#include <iostream> // 引入标准输入输出流库

using namespace cv; // 使用OpenCV命名空间
using namespace std; // 使用标准命名空间

Mat img, inpaintMask; // 声明原图和掩码图像
Point prevPt(-1, -1); // 上一个鼠标点击点，用于绘制线条

// 鼠标回调函数，用于捕捉绘制区域
static void onMouse(int event, int x, int y, int flags, void*) {
    if (event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON))
        prevPt = Point(-1, -1);
    else if (event == EVENT_LBUTTONDOWN)
        prevPt = Point(x, y);
    else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON)) {
        Point pt(x, y);
        if (prevPt.x < 0) prevPt = pt;
        line(inpaintMask, prevPt, pt, Scalar::all(255), 5, 8, 0); // 在掩码图像上绘制线条
        line(img, prevPt, pt, Scalar::all(255), 5, 8, 0); // 在原图上绘制线条
        prevPt = pt;
        imshow("image", img); // 更新显示图像
    }
}

int main() {
    // 硬编码图像路径
    string filename = "..\\images\\image5.png"; // 使用默认图像路径
    img = imread(filename, IMREAD_COLOR); // 读取图像
    if (img.empty()) { // 检查图像是否成功加载
        cout << "Couldn't open the image " << filename << endl;
        return -1;
    }

    namedWindow("image", WINDOW_AUTOSIZE); // 创建显示图像的窗口
    inpaintMask = Mat::zeros(img.size(), CV_8U); // 初始化掩码图像

    imshow("image", img); // 显示原始图像
    setMouseCallback("image", onMouse, NULL); // 设置鼠标回调函数

    while (true) {
        char c = (char)waitKey(); // 等待用户输入

        if (c == 27) // 按ESC键退出程序
            break;

        if (c == 'r') { // 按'r'键恢复原图
            inpaintMask = Scalar::all(0);
            img = imread(filename, IMREAD_COLOR); // 重新加载原图
            imshow("image", img);
        }

        if (c == 'i' || c == ' ') { // 按'i'或空格键执行修复
            Mat inpainted;
            inpaint(img, inpaintMask, inpainted, 3, INPAINT_TELEA); // 执行修复算法
            imshow("inpainted image", inpainted); // 显示修复后的图像
        }
    }

    return 0; // 程序结束
}

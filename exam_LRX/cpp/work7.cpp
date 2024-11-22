//
// Created by liryi on 24-11-22.
// 7.给出图像增强处理中的灰度变换中的线性变换和非线性变换的实例。
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

// 线性变换
Mat linearTransform(const Mat& image, double alpha = 1.5, double beta = 50) {
    Mat result;
    image.convertTo(result, -1, alpha, beta);
    return result;
}

// 对数变换
Mat logTransform(const Mat& image) {
    Mat result, floatImage;
    image.convertTo(floatImage, CV_32F);
    float c = 255 / log(1 + 255); // 常数c
    log(floatImage + 1, result);
    result = result * c;
    result.convertTo(result, CV_8U);
    return result;
}

// 修正后的伽马变换
Mat gammaTransform(const Mat& image, double gamma = 2.0) {
    Mat result;
    unsigned char lookup[256];
    for (int i = 0; i < 256; i++) {
        lookup[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }
    LUT(image, Mat(1, 256, CV_8U, lookup), result);
    return result;
}

// 在图像上添加标识文字
void addLabel(Mat& image, const string& label) {
    putText(image, label, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255), 2);
}

int main() {
    // 读取灰度图像
    string imagePath = "E:\\DL_class\\exam_LRX\\cpp\\images\\image9.png";  // 替换为你的图片路径
    Mat image = imread(imagePath, IMREAD_GRAYSCALE);
    if (image.empty()) {
        cout << "无法加载图像: " << imagePath << endl;
        return -1;
    }

    // 线性变换
    Mat linearImage = linearTransform(image, 1.5, 50);

    // 对数变换
    Mat logImage = logTransform(image);

    // 伽马变换
    Mat gammaImage = gammaTransform(image, 2.0);

    // 添加标识
    addLabel(image, "Original image");
    addLabel(linearImage, "linear transformation");
    addLabel(logImage, "logarithm transformation");
    addLabel(gammaImage, "Gamma transformation");

    // 拼接图片
    Mat topRow, bottomRow, combined;
    hconcat(vector<Mat>{image, linearImage}, topRow);  // 将原图和线性变换拼接
    hconcat(vector<Mat>{logImage, gammaImage}, bottomRow);  // 拼接对数和伽马变换
    vconcat(topRow, bottomRow, combined);  // 上下拼接

    // 显示拼接后的结果
    namedWindow("Comparison of Results", WINDOW_NORMAL);
    imshow("Comparison of Results", combined);

    // 等待用户按下键退出
    waitKey(0);
    destroyAllWindows();
    return 0;
}


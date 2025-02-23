//
// Created by liryi on 24-11-22.
// 10.给出图像处理中频域滤波法中理想低通滤波器、Butterworth低通滤波器、高斯低通滤波器实例各一个。
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

// 频谱中心转换
void shiftDFT(Mat& src, Mat& dst) {
    dst = src.clone();
    int cx = dst.cols / 2;
    int cy = dst.rows / 2;

    Mat q0(dst, Rect(0, 0, cx, cy));
    Mat q1(dst, Rect(cx, 0, cx, cy));
    Mat q2(dst, Rect(0, cy, cx, cy));
    Mat q3(dst, Rect(cx, cy, cx, cy));

    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

// 理想低通滤波器
Mat idealLowpassFilter(Size size, int cutoff) {
    Mat filter(size, CV_32F, Scalar(0));
    Point center(size.width / 2, size.height / 2);

    for (int i = 0; i < size.height; ++i) {
        for (int j = 0; j < size.width; ++j) {
            double distance = sqrt(pow(i - center.y, 2) + pow(j - center.x, 2));
            if (distance <= cutoff) {
                filter.at<float>(i, j) = 1.0;
            }
        }
    }
    return filter;
}

// Butterworth低通滤波器
Mat butterworthLowpassFilter(Size size, int cutoff, int order) {
    Mat filter(size, CV_32F, Scalar(0));
    Point center(size.width / 2, size.height / 2);

    for (int i = 0; i < size.height; ++i) {
        for (int j = 0; j < size.width; ++j) {
            double distance = sqrt(pow(i - center.y, 2) + pow(j - center.x, 2));
            filter.at<float>(i, j) = 1.0 / (1.0 + pow(distance / cutoff, 2 * order));
        }
    }
    return filter;
}

// 高斯低通滤波器
Mat gaussianLowpassFilter(Size size, int cutoff) {
    Mat filter(size, CV_32F, Scalar(0));
    Point center(size.width / 2, size.height / 2);

    for (int i = 0; i < size.height; ++i) {
        for (int j = 0; j < size.width; ++j) {
            double distance = sqrt(pow(i - center.y, 2) + pow(j - center.x, 2));
            filter.at<float>(i, j) = exp(-(pow(distance, 2)) / (2 * pow(cutoff, 2)));
        }
    }
    return filter;
}

// 应用滤波器
Mat applyFilter(const Mat& image, const Mat& filter, Mat& spectrumBefore, Mat& spectrumAfter) {
    Mat padded;
    int m = getOptimalDFTSize(image.rows);
    int n = getOptimalDFTSize(image.cols);
    copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complexImage;
    merge(planes, 2, complexImage);

    // 傅里叶变换
    dft(complexImage, complexImage);

    // 转换到频谱中心
    Mat shifted;
    shiftDFT(complexImage, shifted);

    // 保存原始频谱
    Mat planesSplit[2];
    split(shifted, planesSplit); // 分离实部和虚部
    magnitude(planesSplit[0], planesSplit[1], spectrumBefore); // 计算幅值
    spectrumBefore += Scalar::all(1); // 避免 log(0)
    log(spectrumBefore, spectrumBefore); // 转换到对数尺度

    // 滤波器应用
    planesSplit[0] = planesSplit[0].mul(filter);
    planesSplit[1] = planesSplit[1].mul(filter);
    merge(planesSplit, 2, shifted);

    // 保存滤波后的频谱
    split(shifted, planesSplit);
    magnitude(planesSplit[0], planesSplit[1], spectrumAfter);
    spectrumAfter += Scalar::all(1);
    log(spectrumAfter, spectrumAfter);

    // 逆傅里叶变换
    shiftDFT(shifted, complexImage);
    idft(complexImage, complexImage, DFT_SCALE | DFT_REAL_OUTPUT);
    Mat result;
    complexImage(Rect(0, 0, image.cols, image.rows)).convertTo(result, CV_8U);

    return result;
}

// 在一个窗口中显示原图和滤波后的图像
void displayFilters(const Mat& originalImage, const Mat& idealImage, const Mat& butterworthImage, const Mat& gaussianImage) {
    int width = originalImage.cols;
    int height = originalImage.rows;
    Mat display = Mat::zeros(height, width * 4, originalImage.type());

    originalImage.copyTo(display(Rect(0, 0, width, height)));
    putText(display, "Original", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255), 2);

    idealImage.copyTo(display(Rect(width, 0, width, height)));
    putText(display, "Ideal", Point(width + 10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255), 2);

    butterworthImage.copyTo(display(Rect(width * 2, 0, width, height)));
    putText(display, "Butterworth", Point(width * 2 + 10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255), 2);

    gaussianImage.copyTo(display(Rect(width * 3, 0, width, height)));
    putText(display, "Gaussian", Point(width * 3 + 10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255), 2);

    namedWindow("Filters", WINDOW_NORMAL);
    imshow("Filters", display);
    waitKey(0);
}

int main() {
    Mat image = imread("E:\\DL_class\\exam_LRX\\cpp\\images\\image9.png", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cout << "Error: Image not found!" << endl;
        return -1;
    }

    int cutoff = 50;
    int order = 2;

    Mat idealFilter = idealLowpassFilter(image.size(), cutoff);
    Mat butterworthFilter = butterworthLowpassFilter(image.size(), cutoff, order);
    Mat gaussianFilter = gaussianLowpassFilter(image.size(), cutoff);

    Mat idealResult, butterworthResult, gaussianResult;
    Mat idealSpectrumBefore, idealSpectrumAfter;
    Mat butterworthSpectrumBefore, butterworthSpectrumAfter;
    Mat gaussianSpectrumBefore, gaussianSpectrumAfter;

    idealResult = applyFilter(image, idealFilter, idealSpectrumBefore, idealSpectrumAfter);
    butterworthResult = applyFilter(image, butterworthFilter, butterworthSpectrumBefore, butterworthSpectrumAfter);
    gaussianResult = applyFilter(image, gaussianFilter, gaussianSpectrumBefore, gaussianSpectrumAfter);

    // 显示结果
    displayFilters(image, idealResult, butterworthResult, gaussianResult);

    return 0;
}




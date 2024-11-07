//
// Created by liryi on 24-11-6.
//
#include <iostream>
#include <cstring>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main() {
    auto path = R"(../image/image1.png)";//图片地址
    Mat img = imread(path);
    if (img.empty()) {
        cout << "Error" << endl;
        return -1;
    }
    namedWindow("pic", WINDOW_AUTOSIZE);
    imshow("pic", img);
    waitKey();
    cout << "Hello World!" << endl;
    return 0;
}

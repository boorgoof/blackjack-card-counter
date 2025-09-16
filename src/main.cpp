#include <opencv2/opencv.hpp>
#include <iostream>
int main() {
    cv::Mat img(200, 300, CV_8UC3, cv::Scalar(40, 40, 200));
    cv::putText(img, "Hello, OpenCV!", {20, 110}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {255,255,255}, 2);
    cv::imshow("win", img);
    cv::waitKey(0);
    return 0;
}
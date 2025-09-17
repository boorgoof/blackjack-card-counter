#include <opencv2/opencv.hpp>
#include <iostream>
#include "../include/Label.h"
#include "../include/CardType.h"
#include "../include/Utils.h"

int main() {

    //cv::Mat img(200, 300, CV_8UC3, cv::Scalar(40, 40, 200));
    //cv::putText(img, "Hello, OpenCV!", {20, 110}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {255,255,255}, 2);
    //cv::imshow("win", img);
    //cv::waitKey(0);

    Card_Type card("10S");
    Card_Type card2("10000S");  
    Card_Type card3("AS");  
    Card_Type card4("ASP");  
    std::cout << card << std::endl;
    std::cout << card2 << std::endl;
    std::cout << card3 << std::endl;
    std::cout << card4 << std::endl;
    



    return 0;
}
#include <opencv2/opencv.hpp>
#include <iostream>
#include "../include/Label.h"
#include "../include/CardType.h"
#include "../include/Utils.h"
#include "../include/Loaders.h"


int main() {

    //cv::Mat img(200, 300, CV_8UC3, cv::Scalar(40, 40, 200));
    //cv::putText(img, "Hello, OpenCV!", {20, 110}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {255,255,255}, 2);
    //cv::imshow("win", img);
    //cv::waitKey(0);

    Card_Type card("10S");
    Card_Type card2("10000S");  
    Card_Type card3("AS");  
    Card_Type card4("ASP");
    Card_Type card5("10C");  
    std::cout << card << std::endl;
    std::cout << card2 << std::endl;
    std::cout << card3 << std::endl;
    
    
   
    
    cv::Mat img;
    img = cv::imread("../datasetTest/images/image1.png");
    if (img.empty()) {
        std::cerr << "Could not read the image: " << std::endl;
        return 1;
    }

    std::vector<Label> labels = AnnotationLoaders::load_yolo_image_annotations("../datasetTest/labels/image1.txt", img.cols, img.rows);
    for(int i = 0; i < labels.size(); i++){
        std::cout << labels[i] << std::endl;

        const cv::Rect& rect = labels[i].get_bounding_box();
        cv::rectangle(img, rect, cv::Scalar(0,255,0), 2, cv::LINE_AA);

    }

    cv::imshow("Display window", img);
    cv::waitKey(0);



    



    return 0;
}
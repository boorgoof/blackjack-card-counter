#include <opencv2/opencv.hpp>
#include <iostream>
#include "../include/Label.h"
#include "../include/CardType.h"
#include "../include/Utils.h"
#include "../include/Loaders.h"
#include "../include/StatisticsCalculation.h"


int main() {

    // Test Card_Type
    Card_Type card("10S");
    Card_Type card2("10000S");  
    Card_Type card3("AS");  
    Card_Type card4("ASP");
    Card_Type card5("10C");  
    std::cout << card << std::endl;
    std::cout << card2 << std::endl;
    std::cout << card3 << std::endl;
    
    
   // Test annotation loader
    cv::Mat img;
    img = cv::imread("../data/datasets/single_cards/Images/Images/2C0.jpg");
    if (img.empty()) {
        std::cerr << "Could not read the image: " << std::endl;
        return 1;
    }
    std::cout << img.cols << " " << img.rows << std::endl;

    cv::Mat img2;
    img2 = cv::imread("../data/datasets/single_cards/Images/Images/2C3.jpg");
    if (img.empty()) {
        std::cerr << "Could not read the image: " << std::endl;
        return 1;
    }
    std::cout << img2.cols << " " << img.rows << std::endl;


    std::vector<Label> true_labels = Loader::Annotation::load_yolo_image_annotations("../data/datasets/single_cards/YOLO_Annotations/YOLO_Annotations/2C0.txt", img.cols, img.rows);
    std::vector<Label> pred_labels = Loader::Annotation::load_yolo_image_annotations("../data/datasets/single_cards/YOLO_Annotations/YOLO_Annotations/2C0.txt", img.cols, img.rows);
    std::vector<Label> pred_labels_2 = Loader::Annotation::load_yolo_image_annotations("../data/datasets/single_cards/YOLO_Annotations/YOLO_Annotations/2C3.txt", img2.cols, img2.rows);

    for(int i = 0; i < true_labels.size(); i++){
        
        std::cout << true_labels[i] << std::endl;
        const cv::Rect& rect = true_labels[i].get_bounding_box();
        cv::rectangle(img, rect, cv::Scalar(0,255,0), 2, cv::LINE_AA);

    }           
    cv::imwrite("true_label.png", img);

    for(int i = 0; i < pred_labels_2.size(); i++){
        
        std::cout << pred_labels_2[i] << std::endl;
        const cv::Rect& rect = pred_labels_2[i].get_bounding_box();
        cv::rectangle(img2, rect, cv::Scalar(0,255,0), 2, cv::LINE_AA);

    }
    cv::imwrite("pred_label.png", img2);

    
    // Test StatisticsCalculation
    std::vector<float> meanIoU1= StatisticsCalculation::calc_dataset_meanIoU({true_labels}, {pred_labels});
    std::vector<float> meanIoU2 = StatisticsCalculation::calc_dataset_meanIoU({true_labels}, {pred_labels_2});

    std::cout << "Mean IoU: " << meanIoU1[0] <<  std::endl;
    std::cout << "Mean IoU: " << meanIoU2[0] <<  std::endl;

    return 0;
}
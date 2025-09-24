#include "../../include/card_detector/RoughCardDetector.h"


RoughCardDetector::RoughCardDetector() = default;

std::vector<std::vector<cv::Point>> RoughCardDetector::getCardsPolygon(const cv::Mat& originalImage) {
    std::vector<std::vector<cv::Point>> cards_polygons;
    cv::Mat img = originalImage.clone(); // deep copy
    cv::imshow("Original Image", img);
    // Step 1: Get the white mask
    cv::Mat mask = whiteTreshold(img);
    
    // Step 2: Apply morphological operations to clean up the mask
    cv::Mat kernel_small = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel_small);
    
    cv::Mat kernel_medium = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel_medium);
    
    // Step 3: Filter by size to keep only card-sized objects
    filterBySize(mask, 2000);
    
    cv::imshow("Mask after morphology", mask);

    
    // Apply mask to show result
    img.setTo(cv::Scalar(0, 0, 0), ~mask);
    cv::imshow("Final result", img);
    cv::waitKey(0);

    return cards_polygons;
}

cv::Mat RoughCardDetector::whiteTreshold(const cv::Mat& image) {
    
    cv::Mat enhanced;
    cv::convertScaleAbs(image, enhanced, 1.2, 10); 

    
    cv::Mat hsv;
    cv::cvtColor(enhanced, hsv, cv::COLOR_BGR2HSV);
    
    
    cv::Scalar lower_white(0, 0, 180);     
    cv::Scalar upper_white(179, 20, 255);  
    
    // Create and return mask for white areas
    cv::Mat mask;
    cv::inRange(hsv, lower_white, upper_white, mask);
    
    return mask;
}

void RoughCardDetector::filterBySize(cv::Mat& mask, int minArea) {

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // Create a new mask with only large objects
    cv::Mat filtered_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
    
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        
        // Keep only contours larger than minimum area, to erase text
        if (area >= minArea) {
            cv::fillPoly(filtered_mask, std::vector<std::vector<cv::Point>>{contours[i]}, cv::Scalar(255));
        }
    }
    
    mask = filtered_mask;
}
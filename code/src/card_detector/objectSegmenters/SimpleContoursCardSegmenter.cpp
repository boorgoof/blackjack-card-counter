#include "../../../include/card_detector/objectSegmenters/SimpleContoursCardSegmenter.h"


SimpleContoursCardSegmenter::SimpleContoursCardSegmenter() {
    set_method_name("SimpleContours");
}

std::vector<cv::RotatedRect> SimpleContoursCardSegmenter::segment_objects(
    const cv::Mat& src_img, 
    const cv::Mat& src_mask) {
    
    cv::Mat mask = src_mask.clone();
    
    // Ensure mask is binary
    if (mask.channels() > 1) {
        cv::cvtColor(mask, mask, cv::COLOR_BGR2GRAY);
    }
    cv::threshold(mask, mask, 127, 255, cv::THRESH_BINARY);
    
    // Find external contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask, contours, hierarchy, 
                     cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // Filter contours and create rotated rectangles
    std::vector<cv::RotatedRect> rotatedRects;
    rotatedRects.reserve(contours.size());
    
    for (const auto& contour : contours) {
        // Filter by area and number of points
        double area = cv::contourArea(contour);
        if (area < params_.minCardArea || 
            static_cast<int>(contour.size()) < params_.minContourPoints) {
            continue;
        }
        
        // Fit minimum area rectangle
        cv::RotatedRect rotatedRect = cv::minAreaRect(contour);
        rotatedRects.push_back(rotatedRect);
    }
    
    return rotatedRects;
}
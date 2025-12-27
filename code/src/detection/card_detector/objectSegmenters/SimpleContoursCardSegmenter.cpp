#include "../../../../include/detection/card_detector/objectSegmenters/SimpleContoursCardSegmenter.h"


SimpleContoursCardSegmenter::SimpleContoursCardSegmenter() {
    set_method_name("SimpleContours");
}

std::vector<std::vector<cv::Point>> SimpleContoursCardSegmenter::segment_objects(const cv::Mat& src_img, const cv::Mat& src_mask) {
    
    // find contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(src_mask, contours, hierarchy,  cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // we take all contours founded and we discard that are too small 
    std::vector<std::vector<cv::Point>> filteredContours;
    filteredContours.reserve(contours.size());
    for (const auto& contour : contours) {
        
        double area = cv::contourArea(contour);
        if (area < params_.minCardArea || static_cast<int>(contour.size()) < params_.minContourPoints) {
            continue;
        }
        
        filteredContours.push_back(contour);
    }
    
    return filteredContours;
}
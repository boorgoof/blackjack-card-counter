// Federico Meneghetti
#include "../../../../include/detection/card_detector/objectSegmenters/SimpleContoursCardSegmenter.h"

SimpleContoursCardSegmenter::SimpleContoursCardSegmenter() {
    set_method_name("SimpleContours");
}

std::vector<std::vector<cv::Point>> SimpleContoursCardSegmenter::segment_objects(const cv::Mat& src_img, const cv::Mat& src_mask) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(src_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Discard contours that are too small in area or point count
    std::vector<std::vector<cv::Point>> filteredContours;
    filteredContours.reserve(contours.size());
    for (const std::vector<cv::Point>& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < params_.minCardArea || static_cast<int>(contour.size()) < params_.minContourPoints) {
            continue;
        }
        filteredContours.push_back(contour);
    }

    return filteredContours;
}
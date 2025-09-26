#include "../../include/card_detector/RoughCardDetector.h"


RoughCardDetector::RoughCardDetector() = default;

cv::Mat RoughCardDetector::getCardsMask(const cv::Mat& originalImage) {
    cv::Mat mask = whiteTreshold(originalImage);
    filterBySize(mask, 2000);
    morphologicalCleanup(mask, 5, 9);
    return mask;
}

std::vector<std::vector<cv::Point>> RoughCardDetector::getCardsPolygon(const cv::Mat& originalImage) {
    std::vector<std::vector<cv::Point>> cards_polygons;
    cv::Mat mask = getCardsMask(originalImage);
    cv::findContours(mask, cards_polygons, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    return cards_polygons;
}

std::vector<std::vector<cv::Point>> RoughCardDetector::getConvexHulls(const cv::Mat& originalImage) {
    std::vector<std::vector<cv::Point>> cards_polygons = getCardsPolygon(originalImage);
    std::vector<std::vector<cv::Point>> convex_hulls;
    for (const auto& polygon : cards_polygons) {
        if (!polygon.empty()) {
            std::vector<cv::Point> hull;
            cv::convexHull(polygon, hull);
            convex_hulls.push_back(hull);
        }
    }
    return convex_hulls;
}

std::vector<cv::Rect> RoughCardDetector::getCardsBoundingBox(const cv::Mat& originalImage) {
    std::vector<std::vector<cv::Point>> cards_polygons = getCardsPolygon(originalImage);
    std::vector<cv::Rect> bounding_boxes;
    for (const auto& polygon : cards_polygons) {
        cv::Rect bbox = cv::boundingRect(polygon);
        bounding_boxes.push_back(bbox);
    }
    return bounding_boxes;
}


cv::Mat RoughCardDetector::whiteTreshold(const cv::Mat& image) {
    cv::Mat enhanced;
    cv::convertScaleAbs(image, enhanced, 1.2, 10);
    
    cv::Mat hsv;
    cv::cvtColor(enhanced, hsv, cv::COLOR_BGR2HSV);
    
    cv::Scalar lower_white(0, 0, 180);
    cv::Scalar upper_white(179, 20, 255);
    
    cv::Mat mask;
    cv::inRange(hsv, lower_white, upper_white, mask);
    
    return mask;
}

void RoughCardDetector::filterBySize(cv::Mat& mask, int minArea) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    cv::Mat filtered_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
    
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area >= minArea) {
            cv::fillPoly(filtered_mask, std::vector<std::vector<cv::Point>>{contours[i]}, cv::Scalar(255));
        }
    }
    
    mask = filtered_mask;
}



void RoughCardDetector::morphologicalCleanup(cv::Mat& mask, int openSize, int closeSize) {
    cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(openSize, openSize));
    cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(closeSize, closeSize));
    
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel_open);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel_close);
}
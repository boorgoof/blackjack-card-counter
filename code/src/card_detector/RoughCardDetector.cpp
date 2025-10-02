#include "../../include/card_detector/RoughCardDetector.h"

namespace vision {

RoughCardDetector::RoughCardDetector(PipelinePreset preset) {
    loadPreset(preset);
}

void RoughCardDetector::loadPreset(PipelinePreset preset) {
    clearPipeline();
    
    switch (preset) {
        case PipelinePreset::NONE:
            break;
            
        case PipelinePreset::DEFAULT:
            add_step([](const cv::Mat& in) { return hsvWhiteThreshold(in); });
            add_step([](const cv::Mat& mask) { return filterBySize(mask); });
            add_step([](const cv::Mat& mask) { return morphOpenClose(mask); });
            break;
            
        case PipelinePreset::LOW_LIGHT:
            add_step(hsvWhiteThreshold, cv::Scalar(0,0,80), cv::Scalar(179,60,255), 2.0, 40.0);
            add_step(filterBySize, 2000);
            add_step(morphOpenClose, 5, 9);
            break;
            
        case PipelinePreset::HIGH_LIGHT:
            add_step(hsvWhiteThreshold, cv::Scalar(0,0,140), cv::Scalar(179,40,255), 1.2, 10.0);
            add_step(filterBySize, 2000);
            add_step(morphOpenClose, 5, 9);
            break;
    }
}

cv::Mat RoughCardDetector::getCardPolygonMask(const cv::Mat& img) const {
    cv::Mat current = img.clone();
    for (size_t stepIndex = 0; stepIndex < steps_.size(); ++stepIndex) {
        const std::function<cv::Mat(const cv::Mat&)>& currentStep = steps_[stepIndex];
        cv::Mat processedImage = currentStep(current);
        current = processedImage;
    }
    return current;
}

std::vector<std::vector<cv::Point>> RoughCardDetector::getCardsPolygon(const cv::Mat& img) const {
    cv::Mat mask = getCardPolygonMask(img);
    std::vector<std::vector<cv::Point>> polys;
    cv::findContours(mask, polys, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    return polys;
}

std::vector<std::vector<cv::Point>> RoughCardDetector::getCardsConvexHulls(const cv::Mat& img) const {
    std::vector<std::vector<cv::Point>> polys = getCardsPolygon(img);
    std::vector<std::vector<cv::Point>> hulls;
    hulls.reserve(polys.size());
    for (const std::vector<cv::Point>& p : polys) {
        if (p.empty()) continue;
        std::vector<cv::Point> h;
        cv::convexHull(p, h);
        if (!h.empty()) hulls.push_back(std::move(h));
    }
    return hulls;
}

cv::Mat RoughCardDetector::getCardsConvexHullsMask(const cv::Mat& img) const {
    std::vector<std::vector<cv::Point>> points = getCardsConvexHulls(img);
    cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
    for (const std::vector<cv::Point>& h : points) {
        cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{h}, cv::Scalar(255));
    }
    return mask;
}

std::vector<cv::Rect> RoughCardDetector::getCardsBoundingBox(const cv::Mat& img) const {
    std::vector<std::vector<cv::Point>> polys = getCardsPolygon(img);
    std::vector<cv::Rect> boxes;
    boxes.reserve(polys.size());
    for (const std::vector<cv::Point>& p : polys) {
        cv::Rect boundingBox = cv::boundingRect(p);
        boxes.emplace_back(boundingBox);
    }
    return boxes;
}

cv::Mat RoughCardDetector::getBoundingBoxesMask(const cv::Mat& img) const {
    std::vector<cv::Rect> boxes = getCardsBoundingBox(img);
    cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
    for (const cv::Rect& b : boxes) {
        cv::rectangle(mask, b, cv::Scalar(255), cv::FILLED);
    }
    return mask;
}

}

namespace {

cv::Mat hsvWhiteThreshold(const cv::Mat& bgr, cv::Scalar lo, cv::Scalar hi, double alpha, double beta) {
    cv::Mat enhanced; 
    cv::convertScaleAbs(bgr, enhanced, alpha, beta);
    cv::Mat hsv; 
    cv::cvtColor(enhanced, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask; 
    cv::inRange(hsv, lo, hi, mask);
    return mask;
}

cv::Mat filterBySize(const cv::Mat& maskIn, int minArea) {
    CV_Assert(maskIn.type() == CV_8UC1);
    std::vector<std::vector<cv::Point>> cs;
    cv::findContours(maskIn, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat out = cv::Mat::zeros(maskIn.size(), CV_8UC1);
    for (auto& c : cs) {
        if (cv::contourArea(c) >= minArea) {
            cv::fillPoly(out, std::vector<std::vector<cv::Point>>{c}, cv::Scalar(255));
        }
    }
    return out;
}

cv::Mat morphOpenClose(const cv::Mat& maskIn, int openSize, int closeSize) {
    CV_Assert(maskIn.type() == CV_8UC1);
    cv::Mat mask = maskIn.clone();
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, {openSize, openSize}));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, {closeSize, closeSize}));
    return mask;
}

}

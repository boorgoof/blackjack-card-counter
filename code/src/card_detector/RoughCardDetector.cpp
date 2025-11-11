#include "../../include/card_detector/RoughCardDetector.h"

RoughCardDetector::RoughCardDetector(const PipelinePreset preset, const MaskType maskType) {
    loadPreset(preset);
    loadMaskPreset(maskType);
}

void RoughCardDetector::loadPreset(const PipelinePreset preset) {
    clearPipeline();
    
    switch (preset) {
        case PipelinePreset::NONE:
            break;
            
        case PipelinePreset::DEFAULT:
            add_step(preprocessing::hsvWhiteThreshold, cv::Scalar(0,0,180), cv::Scalar(179,20,255), 1.2, 10.0);
            add_step(preprocessing::filterBySize, 2000);
            add_step(preprocessing::morphOpenClose, 5, 9);
            break;
            
        case PipelinePreset::LOW_LIGHT:
            add_step(preprocessing::hsvWhiteThreshold, cv::Scalar(0,0,80), cv::Scalar(179,60,255), 2.0, 40.0);
            add_step(preprocessing::filterBySize, 2000);
            add_step(preprocessing::morphOpenClose, 5, 9);
            break;
            
        case PipelinePreset::HIGH_LIGHT:
            add_step(preprocessing::hsvWhiteThreshold, cv::Scalar(0,0,140), cv::Scalar(179,40,255), 1.2, 10.0);
            add_step(preprocessing::filterBySize, 2000);
            add_step(preprocessing::morphOpenClose, 5, 9);
            break;
    }
}

void RoughCardDetector::loadMaskPreset(const MaskType maskType) {
    switch (maskType) {
        case MaskType::POLYGON:
            maskType_ = [this](const cv::Mat& img) { 
                return applyPipeline(img); 
            };
            break;
            
        case MaskType::CONVEX_HULL:
            maskType_ = [this](const cv::Mat& img) { 
                cv::Mat processed = applyPipeline(img);
                return mask::getCardsConvexHullsMask(processed); 
            };
            break;
            
        case MaskType::BOUNDING_BOX:
            maskType_ = [this](const cv::Mat& img) { 
                cv::Mat processed = applyPipeline(img);
                return mask::getBoundingBoxesMask(processed); 
            };
            break;
    }
}

cv::Mat RoughCardDetector::applyPipeline(const cv::Mat& img) const {
    cv::Mat current = img.clone();
    for (size_t stepIndex = 0; stepIndex < steps_.size(); ++stepIndex) {
        const std::function<cv::Mat(const cv::Mat&)>& currentStep = steps_[stepIndex];
        cv::Mat processedImage = currentStep(current);
        current = processedImage;
    }
    return current;
}

namespace preprocessing {

cv::Mat hsvWhiteThreshold(const cv::Mat& bgr, const cv::Scalar& lo, const cv::Scalar& hi, double alpha, double beta) {
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
    for (const std::vector<cv::Point>& c : cs) {
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

namespace mask {

std::vector<std::vector<cv::Point>> getCardsPolygon(const cv::Mat& mask) {
    std::vector<std::vector<cv::Point>> polys;
    cv::findContours(mask, polys, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    return polys;
}

std::vector<std::vector<cv::Point>> getCardsConvexHulls(const cv::Mat& mask) {
    std::vector<std::vector<cv::Point>> polys = getCardsPolygon(mask);
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

cv::Mat getCardsConvexHullsMask(const cv::Mat& mask) {
    std::vector<std::vector<cv::Point>> points = getCardsConvexHulls(mask);
    cv::Mat result = cv::Mat::zeros(mask.size(), CV_8UC1);
    for (const std::vector<cv::Point>& h : points) {
        cv::fillPoly(result, std::vector<std::vector<cv::Point>>{h}, cv::Scalar(255));
    }
    return result;
}

std::vector<cv::Rect> getCardsBoundingBox(const cv::Mat& mask) {
    std::vector<std::vector<cv::Point>> polys = getCardsPolygon(mask);
    std::vector<cv::Rect> boxes;
    boxes.reserve(polys.size());
    for (const std::vector<cv::Point>& p : polys) {
        cv::Rect boundingBox = cv::boundingRect(p);
        boxes.emplace_back(boundingBox);
    }
    return boxes;
}

cv::Mat getBoundingBoxesMask(const cv::Mat& mask) {
    std::vector<cv::Rect> boxes = getCardsBoundingBox(mask);
    cv::Mat result = cv::Mat::zeros(mask.size(), CV_8UC1);
    for (const cv::Rect& b : boxes) {
        cv::rectangle(result, b, cv::Scalar(255), cv::FILLED);
    }
    return result;
}

}
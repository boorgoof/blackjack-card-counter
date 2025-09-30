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
            add_step(hsvWhiteThreshold);
            add_step(filterBySize, 2000);
            add_step(morphOpenClose, 5, 9);
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

cv::Mat RoughCardDetector::getCardsMask(const cv::Mat& img) const {
    cv::Mat cur = img;
    for (const auto& step : steps_) cur = step(cur);
    return cur;
}

std::vector<std::vector<cv::Point>>
RoughCardDetector::getCardsPolygon(const cv::Mat& img) const {
    cv::Mat mask = getCardsMask(img);
    std::vector<std::vector<cv::Point>> polys;
    cv::findContours(mask, polys, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    return polys;
}

std::vector<std::vector<cv::Point>>
RoughCardDetector::getConvexHulls(const cv::Mat& img) const {
    auto polys = getCardsPolygon(img);
    std::vector<std::vector<cv::Point>> hulls;
    hulls.reserve(polys.size());
    for (const auto& p : polys) {
        if (p.empty()) continue;
        std::vector<cv::Point> h;
        cv::convexHull(p, h);
        if (!h.empty()) hulls.push_back(std::move(h));
    }
    return hulls;
}

std::vector<cv::Rect>
RoughCardDetector::getCardsBoundingBox(const cv::Mat& img) const {
    auto polys = getCardsPolygon(img);
    std::vector<cv::Rect> boxes;
    boxes.reserve(polys.size());
    for (const auto& p : polys) boxes.emplace_back(cv::boundingRect(p));
    return boxes;
}

}

namespace {

cv::Mat hsvWhiteThreshold(const cv::Mat& bgr, cv::Scalar lo, cv::Scalar hi,
                          double alpha, double beta)
{
    cv::Mat enhanced; 
    cv::convertScaleAbs(bgr, enhanced, alpha, beta);
    cv::Mat hsv; 
    cv::cvtColor(enhanced, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask; 
    cv::inRange(hsv, lo, hi, mask);
    return mask;
}

cv::Mat filterBySize(const cv::Mat& maskIn, int minArea)
{
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

cv::Mat morphOpenClose(const cv::Mat& maskIn, int openSize, int closeSize)
{
    CV_Assert(maskIn.type() == CV_8UC1);
    cv::Mat mask = maskIn.clone();
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, {openSize, openSize}));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, {closeSize, closeSize}));
    return mask;
}

}
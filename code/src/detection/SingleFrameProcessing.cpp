#include "../../include/detection/card_detector/CardProjection.h"
#include "../../include/detection/SingleFrameProcessing.h"
#include "../../include/ObjectType.h"
#include <opencv2/imgproc.hpp>

SingleFrameProcessing::SingleFrameProcessing(std::unique_ptr<CardDetector> card_detector, const bool detect_full_card, const bool visualize)
    : ProcessingMode(detect_full_card, visualize), card_detector_(std::move(card_detector)) {

}

SingleFrameProcessing::~SingleFrameProcessing() {
    
}

std::vector<Label> SingleFrameProcessing::detect_image(const cv::Mat& image) {
    return this->card_detector_->detect_cards(image);
}


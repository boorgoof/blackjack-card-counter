// Gianluca Caregnato
#include "../../include/detection/SequentialFrameProcessing.h"

SequentialFrameProcessing::SequentialFrameProcessing(std::unique_ptr<CardDetector> card_detector, bool visualize, double fps) 
    : ProcessingMode(visualize), card_detector_(std::move(card_detector)), tracker_(fps) {

}

SequentialFrameProcessing::~SequentialFrameProcessing() {}

std::vector<Label> SequentialFrameProcessing::detect_image(const cv::Mat& image) {
    std::vector<Label> detections = card_detector_->detect_cards(image);
    tracker_.update_frame(detections);
    return detections;
}

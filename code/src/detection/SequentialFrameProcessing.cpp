#include "../../include/detection/SequentialFrameProcessing.h"

SequentialFrameProcessing::SequentialFrameProcessing(double fps, bool detect_full_card, bool visualize) 
    : ProcessingMode(detect_full_card, visualize), tracker_(fps) {
    init_detector();
}

SequentialFrameProcessing::~SequentialFrameProcessing() {}

void SequentialFrameProcessing::set_model_path(const std::string& path) {
    model_path_ = path;
    init_detector();
}

void SequentialFrameProcessing::init_detector() {
    card_detector_ = std::make_unique<YoloCardDetector>(model_path_, detect_full_card, visualize);
}

std::vector<Label> SequentialFrameProcessing::detect_image(const cv::Mat& image) {
    std::vector<Label> detections = card_detector_->detect_cards(image);
    tracker_.update_frame(detections);
    return detections;
}

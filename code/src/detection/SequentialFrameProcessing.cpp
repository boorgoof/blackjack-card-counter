#include "../../include/detection/SequentialFrameProcessing.h"

SequentialFrameProcessing::SequentialFrameProcessing(const bool detect_full_card, const bool visualize) : ProcessingMode(detect_full_card, visualize) {
    // Constructor implementation
}

SequentialFrameProcessing::~SequentialFrameProcessing() {
    // Destructor implementation
}

std::vector<Label> SequentialFrameProcessing::detect_image(const cv::Mat& image) {
    std::vector<Label> detected_labels;

    // Image processing and card detection logic goes here

    return detected_labels;
}

#include "../../include/card_detector/SequentialCardDetector.h"

SequentialCardDetector::SequentialCardDetector(bool detect_full_card, bool visualize) : CardDetector(detect_full_card, visualize) {
    // Constructor implementation
}

SequentialCardDetector::~SequentialCardDetector() {
    // Destructor implementation
}

std::vector<Label> SequentialCardDetector::detect_image(const cv::Mat& image) {
    std::vector<Label> detected_labels;

    // Image processing and card detection logic goes here

    return detected_labels;
}

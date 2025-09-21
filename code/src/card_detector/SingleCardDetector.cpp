#include "../../include/card_detector/SingleCardDetector.h"

SingleCardDetector::SingleCardDetector(bool detect_full_card, bool visualize) : CardDetector(detect_full_card, visualize) {
    // Constructor implementation
}

SingleCardDetector::~SingleCardDetector() {
    // Destructor implementation
}

std::vector<Label> SingleCardDetector::detect_image(const cv::Mat& image) {
    std::vector<Label> detected_labels;

    // Perform detection specific to single cards
    // ...

    return detected_labels;
}

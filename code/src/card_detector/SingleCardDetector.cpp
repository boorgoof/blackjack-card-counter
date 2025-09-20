#include "SingleCardDetector.h"

SingleCardDetector::SingleCardDetector(bool detect_full_card) : CardDetector(detect_full_card) {
    // Constructor implementation
}

SingleCardDetector::~SingleCardDetector() {
    // Destructor implementation
}

std::vector<Label> SingleCardDetector::detect_image(const cv::Mat& image, bool is_sequential) {
    std::vector<Label> detected_labels;

    // Perform detection specific to single cards
    // ...

    return detected_labels;
}

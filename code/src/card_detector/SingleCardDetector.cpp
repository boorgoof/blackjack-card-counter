#include "../../include/card_detector/SingleCardDetector.h"

SingleCardDetector::SingleCardDetector(const RoughCardDetector& rough_card_detector, ObjectDetector* object_detector, bool detect_full_card, bool visualize)
    : CardDetector(detect_full_card, visualize), rough_card_detector_(rough_card_detector), object_detector_(object_detector) {
    
}

SingleCardDetector::~SingleCardDetector() {
    
}

std::vector<Label> SingleCardDetector::detect_image(const cv::Mat& image) {
    std::vector<Label> detected_labels;

    //first, use the rough card detector to get a mask of the area where the card is located
    cv::Mat mask = this->rough_card_detector_.getMask(image);

    //then, use the object detector to get the precise location of the card
    if (this->object_detector_) {
        this->object_detector_->detect_objects(image, mask, detected_labels);
    }

    return detected_labels;
}

#include "../../include/card_detector/SingleCardDetector.h"

SingleCardDetector::SingleCardDetector(RoughCardDetector* rough_card_detector, ObjectClassifier* object_classifier, bool detect_full_card, bool visualize)
    : CardDetector(detect_full_card, visualize), rough_card_detector_(rough_card_detector), object_classifier_(object_classifier) {
    
}

SingleCardDetector::~SingleCardDetector() {
    
}

std::vector<Label> SingleCardDetector::detect_image(const cv::Mat& image) {
    std::vector<Label> detected_labels;

    //first, use the rough card detector to get a mask of the area where the card is located
    cv::Mat mask = this->rough_card_detector_->getMask(image);



    if (this->object_classifier_) {
        this->object_classifier_->classify_object(image, mask);
    }

    return detected_labels;
}


#ifndef SINGLE_CARD_DETECTOR_H
#define SINGLE_CARD_DETECTOR_H

#include "CardDetector.h"
#include "RoughCardDetector.h"
#include "objectClassifiers/ObjectClassifier.h"

class SingleCardDetector : public CardDetector {
public:
    SingleCardDetector(RoughCardDetector* rough_card_detector, ObjectClassifier* object_classifier, bool detect_full_card = false, bool visualize = false);
    ~SingleCardDetector();

    std::vector<Label> detect_image(const cv::Mat& image) override;

private:
    std::unique_ptr<RoughCardDetector> rough_card_detector_;
    std::unique_ptr<ObjectClassifier> object_classifier_;
};

#endif
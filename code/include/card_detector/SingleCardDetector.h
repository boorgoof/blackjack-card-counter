#ifndef SINGLE_CARD_DETECTOR_H
#define SINGLE_CARD_DETECTOR_H

#include "CardDetector.h"
#include "RoughCardDetector.h"
#include "objectDetector/ObjectDetector.h"

class SingleCardDetector : public CardDetector {
public:
    SingleCardDetector(const RoughCardDetector& rough_card_detector, ObjectDetector* object_detector, bool detect_full_card = false, bool visualize = false);
    ~SingleCardDetector();

    std::vector<Label> detect_image(const cv::Mat& image) override;

private:
    RoughCardDetector rough_card_detector_;
    std::unique_ptr<ObjectDetector> object_detector_;
};

#endif
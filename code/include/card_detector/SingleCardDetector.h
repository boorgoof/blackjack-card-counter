#ifndef SINGLE_CARD_DETECTOR_H
#define SINGLE_CARD_DETECTOR_H

#include "CardDetector.h"

class SingleCardDetector : public CardDetector {
public:
    SingleCardDetector(bool detect_full_card = false);
    ~SingleCardDetector();

    std::vector<Label> detect_image(const cv::Mat& image) override;
};

#endif
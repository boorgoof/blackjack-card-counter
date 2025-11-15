#ifndef SEQUENTIAL_CARD_DETECTOR_H
#define SEQUENTIAL_CARD_DETECTOR_H

#include "CardDetector.h"

class SequentialCardDetector : public CardDetector {
public:
    SequentialCardDetector(const bool detect_full_card = false, const bool visualize = false);
    ~SequentialCardDetector();
    std::vector<Label> detect_image(const cv::Mat& image) override;
};

#endif

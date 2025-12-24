#ifndef YOLO_CARD_DETECTOR_H
#define YOLO_CARD_DETECTOR_H

#include "CardDetector.h"

class YoloCardDetector : public CardDetector {
public:
    YoloCardDetector(bool detect_full_card = false,bool visualize = false)
        : CardDetector(detect_full_card, visualize) {}

    ~YoloCardDetector() override = default;

    // implementazione nel .cpp
    std::vector<Label> detect_cards(const cv::Mat& image) override;
};

#endif // YOLO_CARD_DETECTOR_H
#ifndef YOLO_CARD_DETECTOR_H
#define YOLO_CARD_DETECTOR_H

#include "CardDetector.h"

class YoloCardDetector : public CardDetector {
public:
    YoloCardDetector(const std::string& modelPath, bool detect_full_card = false, bool visualize = false);

    ~YoloCardDetector() override = default;

    std::vector<Label> detect_cards(const cv::Mat& image) override;

    int mapCardIndex(int inputIndex);

private:
    cv::dnn::Net net;

};

#endif // YOLO_CARD_DETECTOR_H
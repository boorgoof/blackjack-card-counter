// Gianluca Caregnato   

#ifndef SINGLE_CARD_DETECTOR_H
#define SINGLE_CARD_DETECTOR_H

#include "ProcessingMode.h"
#include "card_detector/CardDetector.h"


class SingleFrameProcessing : public ProcessingMode {
public:
    SingleFrameProcessing(std::unique_ptr<CardDetector> card_detector, const bool visualize);
    ~SingleFrameProcessing();

    std::vector<Label> detect_image(const cv::Mat& image) override;
    
private:
    std::unique_ptr<CardDetector> card_detector_;
};

#endif
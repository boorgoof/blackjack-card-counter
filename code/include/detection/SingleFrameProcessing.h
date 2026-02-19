// Gianluca Caregnato

#ifndef SINGLE_CARD_DETECTOR_H
#define SINGLE_CARD_DETECTOR_H

#include "ProcessingMode.h"
#include "card_detector/CardDetector.h"

/**
 * @brief Processes each image independently (no tracking across frames).
 */
class SingleFrameProcessing : public ProcessingMode {
public:
    /**
     * @brief Construct a single-frame processor.
     * @param card_detector Card detector to use (ownership transferred).
     * @param visualize     Whether to visualize detections.
     */
    SingleFrameProcessing(std::unique_ptr<CardDetector> card_detector, bool visualize);
    ~SingleFrameProcessing();

    /**
     * @brief Detect cards in a single image.
     * @param image Input BGR image.
     * @return Labels detected in this image.
     */
    std::vector<Label> detect_image(const cv::Mat& image) override;
    
private:
    std::unique_ptr<CardDetector> card_detector_;
};

#endif
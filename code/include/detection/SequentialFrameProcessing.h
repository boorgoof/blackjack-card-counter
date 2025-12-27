#ifndef SEQUENTIAL_FRAMES_PROCESSING_H
#define SEQUENTIAL_FRAMES_PROCESSING_H

#include "ProcessingMode.h"

class SequentialFrameProcessing : public ProcessingMode {
public:
    SequentialFrameProcessing(const bool detect_full_card = false, const bool visualize = false);
    ~SequentialFrameProcessing();
    std::vector<Label> detect_image(const cv::Mat& image) override;
};

#endif

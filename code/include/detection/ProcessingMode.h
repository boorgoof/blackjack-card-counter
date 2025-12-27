#ifndef PROCESSING_MODE_H
#define PROCESSING_MODE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "../Label.h"

class ProcessingMode {
public:
//delete default constructor, copy constructor and assignment operator
 ProcessingMode(bool detect_full_card = false, bool visualize = false) : detect_full_card(detect_full_card), visualize(visualize) {}
 ProcessingMode(const ProcessingMode&) = delete;
 ProcessingMode& operator=(const ProcessingMode&) = delete;

    virtual  ~ProcessingMode();
    virtual std::vector<Label> detect_image(const cv::Mat& image) = 0;

protected:
    bool detect_full_card;
    bool visualize;
};

#endif // PROCESSING_MODE_H
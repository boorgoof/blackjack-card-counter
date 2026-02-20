// Matteo Bino

#ifndef PROCESSING_MODE_H
#define PROCESSING_MODE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "../Label.h"

class ProcessingMode {
public:
    /**
     * @brief ProcessingMode is an abstract class that defines the interface for different processing modes that can be used to detect cards in images. Each processing mode will have its own implementation of the detect_image method, which takes an image as input and returns a vector of Labels representing the detected cards and their bounding boxes. The visualize flag can be used to indicate whether to visualize the detected cards on the image or not.
     */
    ProcessingMode(bool visualize) : visualize(visualize) {}
    ProcessingMode(const ProcessingMode&) = delete;
    ProcessingMode& operator=(const ProcessingMode&) = delete;

    virtual  ~ProcessingMode();
    virtual std::vector<Label> detect_image(const cv::Mat& image) = 0;

protected:
    bool detect_full_card;
    bool visualize;
};

#endif // PROCESSING_MODE_H
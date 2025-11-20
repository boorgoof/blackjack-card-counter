#ifndef MULTIPLE_CARD_SEGMENTER2_H
#define MULTIPLE_CARD_SEGMENTER2_H

#include "ObjectSegmenter.h"
#include <opencv2/opencv.hpp>

class MultipleCardSegmenter2 : public ObjectSegmenter {
public:
    MultipleCardSegmenter2() = default;
    ~MultipleCardSegmenter2() override = default;

    std::vector<std::vector<cv::Point>> segment_objects(const cv::Mat& image, const cv::Mat& mask) override;
};

#endif // MULTIPLE_CARD_SEGMENTER2_H

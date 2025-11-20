#ifndef MULTIPLE_CARD_SEGMENTER_H
#define MULTIPLE_CARD_SEGMENTER_H

#include "ObjectSegmenter.h"
#include <opencv2/opencv.hpp>

class MultipleCardSegmenter : public ObjectSegmenter {
public:
    MultipleCardSegmenter() = default;
    ~MultipleCardSegmenter() override = default;

    std::vector<std::vector<cv::Point>> segment_objects(const cv::Mat& image, const cv::Mat& mask) override;
};

#endif // MULTIPLE_CARD_SEGMENTER_H

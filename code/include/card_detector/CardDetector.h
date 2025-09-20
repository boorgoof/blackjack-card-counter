#ifndef CARD_DETECTOR_H
#define CARD_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "../Label.h"

class CardDetector {
public:
//delete default constructor, copy constructor and assignment operator
    CardDetector(bool detect_full_card = false) : detect_full_card(detect_full_card) {}
    CardDetector(const CardDetector&) = delete;
    CardDetector& operator=(const CardDetector&) = delete;

    virtual ~CardDetector();

protected:
    bool detect_full_card;
    virtual std::vector<Label> detect_image(const cv::Mat& image, bool is_sequential) = 0;
};

#endif // CARD_DETECTOR_H
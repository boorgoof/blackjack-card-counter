#ifndef CARD_DETECTOR_H
#define CARD_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "../../Label.h"

class CardDetector {
public:
//delete default constructor, copy constructor and assignment operator
 CardDetector(bool detect_full_card = false, bool visualize = false) : detect_full_card(detect_full_card), visualize(visualize) {}
 CardDetector(const CardDetector&) = delete;
 CardDetector& operator=(const CardDetector&) = delete;

    virtual  ~CardDetector();
    virtual std::vector<Label> detect_cards(const cv::Mat& image) = 0;

protected:
    bool detect_full_card;
    bool visualize;
};

#endif // CARD_DETECTOR_H
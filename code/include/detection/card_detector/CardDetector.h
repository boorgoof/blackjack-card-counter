//Matteo Bino

#ifndef CARD_DETECTOR_H
#define CARD_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "../../Label.h"

class CardDetector {
public:
    /**
     * Abstract base class for card detectors. It defines the interface for detecting cards in an image, and it also contains a flag to visualize the detected cards on the image (for development purposes).
     */
    CardDetector(bool visualize) : visualize(visualize) {}
    CardDetector(const CardDetector&) = delete;
    CardDetector& operator=(const CardDetector&) = delete;

virtual  ~CardDetector();

/**
 * @brief detect cards in an image. It is virtual and should be implemented by the specific card detector classes (e.g. SegmentationClassificationCardDetector, YoloCardDetector, DistanceTrasformCardDetector..)
 * @param image the image to detect cards in
 * @return a vector of Labels, where each Label contains the detected card and its bounding box
 */
virtual std::vector<Label> detect_cards(const cv::Mat& image) = 0;

protected:
    bool visualize;
};

#endif // CARD_DETECTOR_H
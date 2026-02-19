// Gianluca Caregnato

#ifndef SIMPLE_CONTOURS_CARD_SEGMENTER_H
#define SIMPLE_CONTOURS_CARD_SEGMENTER_H

#include "ObjectSegmenter.h"
#include <opencv2/opencv.hpp>

/**
 * @brief Segmenter that finds card contours directly from a binary mask.
 *
 * Uses cv::findContours on the mask and filters out contours
 * that are too small in area or have too few points.
 */
class SimpleContoursCardSegmenter : public ObjectSegmenter {
private:
    struct Params {
        double minCardArea = 1000.0;
        int minContourPoints = 5;
    };
    Params params_;

public:
    SimpleContoursCardSegmenter();
    ~SimpleContoursCardSegmenter() override = default;

    /**
     * @brief Segment card-shaped objects from a binary mask.
     * @param src_img  Source image (unused — segmentation relies only on the mask).
     * @param src_mask Binary mask whose contours are extracted.
     * @return Filtered contours, each a vector of boundary points.
     */
    std::vector<std::vector<cv::Point>> segment_objects(const cv::Mat& src_img, const cv::Mat& src_mask) override;

    void setMinCardArea(double area) { params_.minCardArea = area; }
    void setMinContourPoints(int points) { params_.minContourPoints = points; }

    double getMinCardArea() const { return params_.minCardArea; }
    int getMinContourPoints() const { return params_.minContourPoints; }
};

#endif // SIMPLE_CONTOURS_CARD_SEGMENTER_H
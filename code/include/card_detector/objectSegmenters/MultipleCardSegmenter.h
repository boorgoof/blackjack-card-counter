#ifndef MULTIPLE_CARD_SEGMENTER_H
#define MULTIPLE_CARD_SEGMENTER_H

#include "ObjectSegmenter.h"
#include <opencv2/opencv.hpp>

class MultipleCardSegmenter : public ObjectSegmenter {
private:
    struct Params {
        double minCardArea = 1000.0;
        int morphKernelSize = 5;
        int erosionIterations = 2;
        int dilationIterations = 2;
        float distThresholdPercent = 0.5f;
        int minDistanceBetweenCards = 30;
        int maxCards = 50;
    };

    Params params_;
    
    std::vector<cv::Point> findCardCenters(const cv::Mat& dist, float threshold, int minDistance);
    std::vector<cv::Point> findContourAroundCenter(const cv::Mat& mask, const cv::Point& center, int searchRadius);

public:
    MultipleCardSegmenter();
    ~MultipleCardSegmenter() override = default;

    std::vector<std::vector<cv::Point>> segment_objects(const cv::Mat& src_img, const cv::Mat& src_mask) override;

    void setMinCardArea(double area) { params_.minCardArea = area; }
    void setMorphKernelSize(int size) { params_.morphKernelSize = size; }
    void setErosionIterations(int iter) { params_.erosionIterations = iter; }
    void setDilationIterations(int iter) { params_.dilationIterations = iter; }
    void setDistThresholdPercent(float percent) { params_.distThresholdPercent = percent; }
    void setMinDistanceBetweenCards(int dist) { params_.minDistanceBetweenCards = dist; }
    void setMaxCards(int max) { params_.maxCards = max; }

    double getMinCardArea() const { return params_.minCardArea; }
    int getMorphKernelSize() const { return params_.morphKernelSize; }
    int getErosionIterations() const { return params_.erosionIterations; }
    int getDilationIterations() const { return params_.dilationIterations; }
    float getDistThresholdPercent() const { return params_.distThresholdPercent; }
    int getMinDistanceBetweenCards() const { return params_.minDistanceBetweenCards; }
    int getMaxCards() const { return params_.maxCards; }
};

#endif // MULTIPLE_CARD_SEGMENTER_H

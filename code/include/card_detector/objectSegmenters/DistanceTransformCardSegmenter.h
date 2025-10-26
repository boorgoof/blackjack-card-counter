#ifndef DISTANCE_TRANSFORM_CARD_SEGMENTER_H
#define DISTANCE_TRANSFORM_CARD_SEGMENTER_H

#include "ObjectSegmenter.h"
#include <opencv2/opencv.hpp>


class DistanceTransformCardSegmenter : public ObjectSegmenter {
private:

    struct Params {
        // Distance transform parameters
        float distThresholdPercent = 0.7f;     // Fraction of max distance value to consider as peak
        int fixedMinDistance = 50;              // Minimum distance between detected card centers
        
        // Local region extraction
        float roiSizeMultiplier = 1.2f;        // Multiplier for ROI size around each peak
        int minContourPoints = 5;               // Minimum contour points for local detection
        
        // Fallback rectangle dimensions
        float bboxWidthMultiplier = 1.8f;      // Width multiplier if no contour found
        float bboxHeightMultiplier = 2.5f;     // Height multiplier if no contour found
        
        // Safety limits
        int maxCards = 50;                     // Maximum number of cards to detect
    };

    Params params_;
    std::vector<cv::Point> findLocalMaxima(const cv::Mat& dist, float threshold, int minDistance);

public:
    
    DistanceTransformCardSegmenter();
    ~DistanceTransformCardSegmenter() override = default;

    
    std::vector<cv::RotatedRect> segment_objects(const cv::Mat& src_img, const cv::Mat& src_mask) override;

    
    void setDistThresholdPercent(float percent) { params_.distThresholdPercent = percent; }
    void setFixedMinDistance(int dist) { params_.fixedMinDistance = dist; }
    void setROISizeMultiplier(float mult) { params_.roiSizeMultiplier = mult; }
    void setMinContourPoints(int points) { params_.minContourPoints = points; }
    void setBBoxWidthMultiplier(float mult) { params_.bboxWidthMultiplier = mult; }
    void setBBoxHeightMultiplier(float mult) { params_.bboxHeightMultiplier = mult; }
    void setMaxCards(int maxCards) { params_.maxCards = maxCards; }

    float getDistThresholdPercent() const { return params_.distThresholdPercent; }
    int getFixedMinDistance() const { return params_.fixedMinDistance; }
    float getROISizeMultiplier() const { return params_.roiSizeMultiplier; }
    int getMinContourPoints() const { return params_.minContourPoints; }
    float getBBoxWidthMultiplier() const { return params_.bboxWidthMultiplier; }
    float getBBoxHeightMultiplier() const { return params_.bboxHeightMultiplier; }
    int getMaxCards() const { return params_.maxCards; }
};

#endif // DISTANCE_TRANSFORM_CARD_SEGMENTER_H

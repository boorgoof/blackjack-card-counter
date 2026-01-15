#ifndef KEYPOINT_EXTRACTOR_H
#define KEYPOINT_EXTRACTOR_H

#include <opencv2/opencv.hpp>
#include "FeatureExtractor.h"

class KeypointExtractor : public FeatureExtractor {
private:
    /**
     * @brief the OpenCV feature Extractor
     */
    cv::Ptr<cv::Feature2D> features_extractor;
    
    void init();

public:
    KeypointExtractor(const ExtractorType::FeatureDescriptorAlgorithm& type) : FeatureExtractor(type) {this->init();}
    ~KeypointExtractor();
    
    Feature* extractFeatures(const cv::Mat& img, const cv::Mat& mask) const;
};

#endif
//Matteo Bino

#ifndef HASH_EXTRACTOR_H
#define HASH_EXTRACTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/img_hash.hpp>
#include "../features/HashFeature.h"
#include "FeatureExtractor.h"

/**
 * @brief HashExtractor class to extract hash features from images.
 *  
 */
class HashExtractor : public FeatureExtractor {
public:
    HashExtractor(const ExtractorType::FeatureDescriptorAlgorithm& type) : FeatureExtractor(type) {this->init();}
    
    /**
     * @brief extract features from an image given a mask to select the region of interest.
     */
    Feature* extractFeatures(const cv::Mat& img, const cv::Mat& mask = cv::Mat()) const override;
private:

    /**
     * @brief the OpenCV hash extractor
     */
    cv::Ptr<cv::img_hash::ImgHashBase> hasher_;

    void init();
};

#endif // HASH_EXTRACTOR_H
#include "../../../../include/card_detector/ObjectDetector/FeaturePipeline/FeatureExtractor.h"

FeatureExtractor::~FeatureExtractor() {
    this->features_extractor.release();
}

void FeatureExtractor::init(){
    switch (this->extractor_type) {
        case ExtractorType::FeatureDescriptorAlgorithm::SIFT:
            this->features_extractor = cv::SIFT::create();
            break;
        case ExtractorType::FeatureDescriptorAlgorithm::ORB:
            this->features_extractor = cv::ORB::create();
            break;
        default:
            throw std::invalid_argument("Invalid featureExtractor type");
    }
}

void FeatureExtractor::extractFeatures(const cv::Mat& img, const cv::Mat& mask, KeypointFeature& feature) const {
    
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    this->features_extractor->detectAndCompute(img, mask.empty() ? cv::noArray() : mask, keypoints, descriptors);

    feature.setKeypoints(keypoints);
    feature.setDescriptors(descriptors);
}


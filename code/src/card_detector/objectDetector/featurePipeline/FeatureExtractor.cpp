#include "../../../../include/card_detector/ObjectDetector/FeaturePipeline/FeatureExtractor.h"

FeatureExtractor::~FeatureExtractor() {
    this->features_extractor.release();
}

void FeatureExtractor::init(){
    switch (this->type) {
        case ExtractorType::Type::SIFT:
            this->features_extractor = cv::SIFT::create();
            break;
        case ExtractorType::Type::ORB:
            this->features_extractor = cv::ORB::create();
            break;
        default:
            throw std::invalid_argument("Invalid featureExtractor type");
    }
}


void FeatureExtractor::extractFeatures(const cv::Mat& img, const cv::Mat& mask, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const {
    
    
    keypoints.clear(); // clear the output vector of keypoints we don't want computeOnly
    // descriptors.release(); detectAndCompute overwrites them
    
    this->features_extractor->detectAndCompute(img, mask.empty() ? cv::noArray() : mask, keypoints, descriptors);
}


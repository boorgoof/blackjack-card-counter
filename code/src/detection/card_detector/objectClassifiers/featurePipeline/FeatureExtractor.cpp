#include "../../../../../include/detection/card_detector/objectClassifiers/featurePipeline/FeatureExtractor.h"

FeatureExtractor::~FeatureExtractor() {
    this->features_extractor.release();
}

void FeatureExtractor::init(){
    switch (this->extractor_type) {
        case ExtractorType::FeatureDescriptorAlgorithm::SIFT:
            this->features_extractor = cv::SIFT::create(); //3500, 3, 0.02, 12, 1.6
            break;
        case ExtractorType::FeatureDescriptorAlgorithm::ORB:
            this->features_extractor = cv::ORB::create();
            break;
        default:
            throw std::invalid_argument("Invalid featureExtractor type");
    }
}

Feature* FeatureExtractor::extractFeatures(const cv::Mat& img, const cv::Mat& mask) const {
    
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    this->features_extractor->detectAndCompute(img, mask.empty() ? cv::noArray() : mask, keypoints, descriptors);

    std::vector<cv::Point2f> rect_points = {cv::Point2f(0,0), cv::Point2f(static_cast<float>(img.cols-1),0), cv::Point2f(static_cast<float>(img.cols-1),static_cast<float>(img.rows-1)), cv::Point2f(0,static_cast<float>(img.rows-1))};
    return new KeypointFeature(keypoints, descriptors, rect_points);
    
}


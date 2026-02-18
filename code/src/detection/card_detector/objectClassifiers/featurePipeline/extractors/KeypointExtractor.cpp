// Federico Meneghetti

#include "../../../../../../include/detection/card_detector/objectClassifiers/featurePipeline/extractors/KeypointExtractor.h"
#include "../../../../../../include/detection/card_detector/objectClassifiers/featurePipeline/features/KeypointFeature.h"

KeypointExtractor::~KeypointExtractor() {
    this->features_extractor.release();
}

void KeypointExtractor::init(){
    switch (this->type_) {
        case ExtractorType::FeatureDescriptorAlgorithm::SIFT:
            this->features_extractor = cv::SIFT::create(1200, 3, 0.02, 8, 1.2);
            break;
        case ExtractorType::FeatureDescriptorAlgorithm::ORB:
            this->features_extractor = cv::ORB::create();
            break;
        default:
            throw std::invalid_argument("Invalid KeypointExtractor type");
    }
}

Feature* KeypointExtractor::extractFeatures(const cv::Mat& img, const cv::Mat& mask) const {
    
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    this->features_extractor->detectAndCompute(img, mask.empty() ? cv::noArray() : mask, keypoints, descriptors);

    std::vector<cv::Point2f> rect_points = {cv::Point2f(0,0), cv::Point2f(static_cast<float>(img.cols-1),0), cv::Point2f(static_cast<float>(img.cols-1),static_cast<float>(img.rows-1)), cv::Point2f(0,static_cast<float>(img.rows-1))};
    return new KeypointFeature(keypoints, descriptors, rect_points);
    
}


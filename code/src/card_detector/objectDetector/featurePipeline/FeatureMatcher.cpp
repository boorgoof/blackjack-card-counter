#include "../../../../include/card_detector/objectDetector/featurePipeline/FeatureMatcher.h"


FeatureMatcher::~FeatureMatcher() {
    this->features_matcher.release();
}

void FeatureMatcher::init(){
    switch (this->type) {
        case MatcherType::MatcherAlgorithm::FLANN:
            this->features_matcher = cv::FlannBasedMatcher::create();
            break;
        case MatcherType::MatcherAlgorithm::BRUTEFORCE_HAMMING:
            this->features_matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
            break;
        default: 
           throw std::invalid_argument("Invalid featureMatcher type");
    }
}

void FeatureMatcher::matchFeatures( const cv::Mat& modelDescriptors, const cv::Mat& sceneDescriptors, std::vector<cv::DMatch>& matches) const{
    
    matches.clear();

    if (modelDescriptors.empty()) {
        throw std::invalid_argument("Model descriptors are empty during feature matching");
    }
    if (sceneDescriptors.empty()) {
        throw std::invalid_argument("Scene descriptors are empty during feature matching");
    }

    std::vector<std::vector<cv::DMatch>> knn_matches;
    this->features_matcher->knnMatch(modelDescriptors, sceneDescriptors, knn_matches, 2);  
    
    //apply Lowe's ratio test
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i].size() >= 2) {
            const cv::DMatch& m = knn_matches[i][0];
            const cv::DMatch& n = knn_matches[i][1];
            if (m.distance < this->lowe_ratio_thresh * n.distance ) { 
                matches.push_back(m);
            }
        }
    }

}
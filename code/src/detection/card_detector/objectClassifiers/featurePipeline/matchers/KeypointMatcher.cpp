#include "../../../../../../include/detection/card_detector/objectClassifiers/featurePipeline/matchers/KeypointMatcher.h"
#include "../../../../../../include/detection/card_detector/objectClassifiers/featurePipeline/features/KeypointFeature.h"

KeypointMatcher::~KeypointMatcher() {
    this->features_matcher.release();
}

void KeypointMatcher::init(){
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

double KeypointMatcher::matchFeatures(const Feature* query, const Feature* target) const{
    
    auto q = dynamic_cast<const KeypointFeature*>(query);
    auto t = dynamic_cast<const KeypointFeature*>(target);
    if (!q || !t || q->getDescriptors().empty() || t->getDescriptors().empty()) return 0.0;

    std::vector<std::vector<cv::DMatch>> knn_matches;
    this->features_matcher->knnMatch(q->getDescriptors(), t->getDescriptors(), knn_matches, 2);
    
    //apply Lowe's ratio test to select good matches
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i].size() >= 2) {
            const cv::DMatch& m = knn_matches[i][0];
            const cv::DMatch& n = knn_matches[i][1];
            if (m.distance < lowe_ratio_thresh_ * n.distance) { 
                good_matches.push_back(m);
            }
        }
    }
    
    return good_matches.size();
}

double KeypointMatcher::calculateRatio(double best, double second_best) const
{
    //scores go from 0 (worst) to +inf (best)
    if (second_best == 0.0) {
        return std::numeric_limits<double>::max(); // Avoid division by zero
    }
    return second_best / best;
}
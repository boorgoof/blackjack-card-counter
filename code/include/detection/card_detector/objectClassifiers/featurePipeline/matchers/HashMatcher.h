//Matteo Bino

#ifndef HASH_MATCHER_H
#define HASH_MATCHER_H

#include <opencv2/opencv.hpp>
#include <opencv2/img_hash.hpp>
#include "../features/Feature.h"
#include "FeatureMatcher.h"


/**
 * @brief HashMatcher class to match hash features between two images
 */
class HashMatcher : public FeatureMatcher {

public:
    HashMatcher(const MatcherType::MatcherAlgorithm& type, double threshold);
    double matchFeatures(const Feature* query, const Feature* target) const override;
    
    bool isValid(double score) const override { return score <= threshold_; }
    bool isBetter(double score1, double score2) const override { return score1 < score2; }
    
    double getWorstScore() const override { return std::numeric_limits<double>::max(); }
    double calculateRatio(double best, double second_best) const override;
private:
    cv::Ptr<cv::img_hash::ImgHashBase> hasher_;
    double threshold_;
};

#endif // HASH_MATCHER_H
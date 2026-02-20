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
    /**
     * HashMatcher uses the OpenCV img_hash module to compute the hash of the input images and compare them using the compare method of the hasher. The specific type of hash used is determined by the type parameter passed to the constructor, which can be one of the following: PHASH or COLOR_MOMENT_HASH. The threshold parameter is used to determine if a given score is valid (if it is good enough) and to compare scores.
     * @param type the type of hash to use for matching (PHASH or COLOR_MOMENT_HASH)
     * @param threshold the threshold to use for determining if a score is valid and for comparing scores
     */
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
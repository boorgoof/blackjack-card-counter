#ifndef KEYPOINT_MATCHER_H
#define KEYPOINT_MATCHER_H

#include <opencv2/opencv.hpp>
#include "../features/Feature.h"
#include "FeatureMatcher.h"

class KeypointMatcher : public FeatureMatcher {

private:
    cv::Ptr<cv::DescriptorMatcher> features_matcher;
    size_t min_matches_threshold_ = 10;
    float lowe_ratio_thresh_ = 0.8;
    void init();

public:
    KeypointMatcher(const MatcherType::MatcherAlgorithm& type, size_t min_matches_threshold = 10, float lowe_ratio_thresh = 0.8) : FeatureMatcher(type), min_matches_threshold_{min_matches_threshold}, lowe_ratio_thresh_{lowe_ratio_thresh} {this->init();}
    //destructor
    ~KeypointMatcher();

    double matchFeatures(const Feature* target, const Feature* query) const override;
    bool isValid(double score) const override {return score >= static_cast<double>(min_matches_threshold_);}
    bool isBetter(double score1, double score2) const override {return score1 > score2;}
    double getWorstScore() const override {return 0.0;}
    double calculateRatio(double best, double second_best) const override;
};

#endif // KEYPOINT_MATCHER_H
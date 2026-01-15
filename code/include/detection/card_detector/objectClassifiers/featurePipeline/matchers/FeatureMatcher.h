#ifndef FEATURE_MATCHER_H
#define FEATURE_MATCHER_H

#include <opencv2/opencv.hpp>
#include "../features/Feature.h"

class MatcherType {
public:

    /**
     * @brief enum to represent the different types of matchers.
     */
    enum MatcherAlgorithm {
        BRUTEFORCE_HAMMING,
        FLANN,
        PHASH,
        COLOR_MOMENT_HASH
    };

    static std::string toString(MatcherAlgorithm type) {
        switch (type) {
            case BRUTEFORCE_HAMMING: return "BRUTEFORCE_HAMMING";
            case FLANN: return "FLANN";
            case PHASH: return "PHASH";
            case COLOR_MOMENT_HASH: return "COLOR_MOMENT_HASH";
            default: throw std::invalid_argument("Unknown Matcher type");
        }
    }

private:
    /**
     * @brief the type of the Matcher
     */
    MatcherAlgorithm type;
};

class FeatureMatcher {
public:
    FeatureMatcher(const MatcherType::MatcherAlgorithm& type) : type{type} {}
    virtual ~FeatureMatcher() = default;
    virtual double matchFeatures(const Feature* query, const Feature* target) const = 0;
    virtual bool isValid(double score) const = 0;
    virtual bool isBetter(double score1, double score2) const = 0;
    virtual double getWorstScore() const = 0;
    virtual double calculateRatio(double best, double second_best) const = 0;

    MatcherType::MatcherAlgorithm getType() const {return type;}
    void setType(const MatcherType::MatcherAlgorithm& type) {this->type = type;}
protected:
    MatcherType::MatcherAlgorithm type;
};

#endif // FEATURE_MATCHER_H
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

    /**
     * @brief match features between two images (the query and the target)
     * @param query the feature of the query (model) image
     * @param target the feature of the target (scene) image
     * @return a score representing how well the features match
     */
    virtual double matchFeatures(const Feature* query, const Feature* target) const = 0;

     /**
     * @brief check if a given score is valid (if is good enough )
     * @param score the score to check
     * @return true if the score is valid, false otherwise
     */
    virtual bool isValid(double score) const = 0;

    /**
     * @brief check if a given score is better than another score
     * @param score1 the first score
     * @param score2 the second score
     * @return true if score1 is better than score2, false otherwise
     */ 
    virtual bool isBetter(double score1, double score2) const = 0;

    /**
     * @brief get the worst possible score for this matcher
     * @return the worst possible score
     */
    virtual double getWorstScore() const = 0;

    /**
     * @brief calculate a ratio between the best score and the second best score
     * @param best the best score
     * @param second_best the second best score
     * @return the calculated ratio
     */ 
    virtual double calculateRatio(double best, double second_best) const = 0;

     /**
     * @brief get the type of the matcher
     * @return the type of the matcher
     */ 
    MatcherType::MatcherAlgorithm getType() const {return type;}

    /**
    * @brief set the type of the matcher
    * @param type the type to set
    */
    void setType(const MatcherType::MatcherAlgorithm& type) {this->type = type;}

protected:
    MatcherType::MatcherAlgorithm type;
};

#endif // FEATURE_MATCHER_H
#ifndef FEATURE_MATCHER_H
#define FEATURE_MATCHER_H

#include <opencv2/opencv.hpp>

/**
 * @brief Class to represent the type of a matcher (e.g. FLANN, BRUTEFORCE).
 */
class MatcherType{
public:

    /**
     * @brief enum to represent the different types of matchers.
     */
    enum MatcherAlgorithm{
        FLANN,
        BRUTEFORCE_HAMMING
    };

    static std::vector<MatcherAlgorithm> getMatcherTypes() {
        return {MatcherAlgorithm::FLANN,
                MatcherAlgorithm::BRUTEFORCE_HAMMING};
    }

   
    static std::string toString(MatcherAlgorithm type) {
        switch (type) {
            case MatcherAlgorithm::FLANN: return "FLANN";
            case MatcherAlgorithm::BRUTEFORCE_HAMMING: return "BRUTEFORCE_HAMMING";
            default: throw std::invalid_argument("No matcher type");
        }
    }
       
private:
    /**
     * @brief the type of the matcher
     */
    MatcherAlgorithm type;

};

class FeatureMatcher{

private:
    /**
     * @brief the type of the matcher
     */
    MatcherType::MatcherAlgorithm type;
    /**
     * @brief the OpenCV feature matcher
     */
    cv::Ptr<cv::DescriptorMatcher> features_matcher;
    float lowe_ratio_thresh;

    void init();

public:
    FeatureMatcher(const MatcherType::MatcherAlgorithm& type, float ratio_thresh = 0.8f) : type{type}, lowe_ratio_thresh{ratio_thresh} {this->init();}
    FeatureMatcher(const MatcherType::MatcherAlgorithm& type, cv::DescriptorMatcher* matcher, float ratio_thresh = 0.8f) : type{type}, features_matcher{cv::Ptr<cv::DescriptorMatcher>(matcher)}, lowe_ratio_thresh{ratio_thresh} {}
    //destructor
    ~FeatureMatcher();

    /**
     * @brief match features between two images (the model and the scene)
     * @param modelDescriptors the descriptors of the model image
     * @param sceneDescriptors the descriptors of the scene image
     * @param matches the output vector of matches between the two input images
     */
    void matchFeatures(const cv::Mat& modelDescriptors, const cv::Mat& sceneDescriptors, std::vector<cv::DMatch>& matches) const;
    
    void updateMatcher(cv::Ptr<cv::DescriptorMatcher> new_matcher) {
        this->features_matcher.release();
        this->features_matcher = new_matcher;
    }

    const MatcherType::MatcherAlgorithm& getType() const {return type;}
    void setType(const MatcherType::MatcherAlgorithm& type) {this->type = type;}

    float getLoweRatioThresh() const { return lowe_ratio_thresh; }
    void setLoweRatioThresh(float t) {
        if (t <= 0.0f || !std::isfinite(t))
            throw std::invalid_argument("lowe_ratio_thresh must be > 0");
        lowe_ratio_thresh = t;
    }

   
};

#endif // FEATURE_MATCHER_H
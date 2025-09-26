//Federico Meneghetti

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
    enum class Type{
        FLANN,
        BRUTEFORCE
    };

    static std::vector<MatcherType::Type> getMatcherTypes() {
        return { MatcherType::Type::FLANN, /*MatcherType::Type::BRUTEFORCE*/ };
    }

   
    static std::string toString(Type type) {
        switch (type) {
            case Type::FLANN: return "FLANN";
            case Type::BRUTEFORCE: return "BRUTEFORCE";
            default: throw std::invalid_argument("No matcher type");
        }
    }
       
    private:
    /**
     * @brief the type of the matcher
     */
    Type type;

};
class FeatureMatcher{
    private:
    /**
     * @brief the type of the matcher
     */
    MatcherType::Type type;
    /**
     * @brief the OpenCV feature matcher
     */
    cv::Ptr<cv::DescriptorMatcher> features_matcher;
    void init();

    public:
    FeatureMatcher(const MatcherType::Type& type) : type{type} {this->init();}
    FeatureMatcher(const MatcherType::Type& type, cv::DescriptorMatcher* matcher) : type{type}, features_matcher{cv::Ptr<cv::DescriptorMatcher>(matcher)} {}
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

    const MatcherType::Type& getType() const {return type;}
    void setType(const MatcherType::Type& type) {this->type = type;}

   
};

#endif
#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <opencv2/opencv.hpp>
class ImageFilter;

/**
 * @brief Class to represent the type of a extractor (e.g. SIFT, ORB, SURF).
 */
class ExtractorType{

    public:

    /**
     * @brief enum to represent the different types of extractors.
     */
    enum class Type{
        SIFT,
        ORB
    };

    static std::vector<ExtractorType::Type> getExtractorTypes() {
        return { 
            ExtractorType::Type::SIFT,
            ExtractorType::Type::ORB};
    }

   
    static std::string toString(Type type) {
        switch (type) {
            case Type::SIFT: return "SIFT";
            case Type::ORB: return "ORB";
            default: throw std::invalid_argument("Unknown Extractor type");
        }
    }
           
    private:
    /**
     * @brief the type of the Extractor
     */
    Type type;

};

class FeatureExtractor{

    private:
    /**
     * @brief the type of the Extractor
     */
    ExtractorType::Type type;

    /**
     * @brief the OpenCV feature Extractor
     */
    cv::Ptr<cv::Feature2D> features_extractor;

    void init();

    public:
    FeatureExtractor(const ExtractorType::Type& type) : type{type} {this->init();}
    ~FeatureExtractor();
    
    /**
     * @brief extract features of an image
     * @param img the image to extract features from
     * @param mask the mask that select the areas of the image to extract keypoints from
     * @param keypoints the output vector of keypoints
     * @param descriptors the output matrix of descriptors
     */
    void extractFeatures(const cv::Mat& img, const cv::Mat& mask, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const;
    
     

    void updateExtractor(cv::Ptr<cv::Feature2D> new_extractor) {
        this->features_extractor.release();
        this->features_extractor = new_extractor;
    }

    const ExtractorType::Type& getType() const {return type;}
    void setType(const ExtractorType::Type& type) {this->type = type;}

    

};

#endif
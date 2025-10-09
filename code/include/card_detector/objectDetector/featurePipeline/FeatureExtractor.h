#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <opencv2/opencv.hpp>
#include "features/KeypointFeature.h"
class ImageFilter;

/**
 * @brief Class to represent the type of a extractor (e.g. SIFT, ORB, SURF).
 */
class ExtractorType{

    public:

    /**
     * @brief enum to represent the different types of extractors.
     */
    enum FeatureDescriptorAlgorithm{
        SIFT,
        ORB,
        HASH
    };

    static std::vector<FeatureDescriptorAlgorithm> getExtractorTypes() {
        return { 
            ExtractorType::SIFT,
            ExtractorType::ORB};
    }

   
    static std::string toString(FeatureDescriptorAlgorithm type) {
        switch (type) {
            case SIFT: return "SIFT";
            case ORB: return "ORB";
            default: throw std::invalid_argument("Unknown Extractor type");
        }
    }
           
    private:
    /**
     * @brief the type of the Extractor
     */
    FeatureDescriptorAlgorithm type;

};

class FeatureExtractor{

    private:
    /**
     * @brief the type of the Extractor
     */
    ExtractorType::FeatureDescriptorAlgorithm extractor_type;

    /**
     * @brief the OpenCV feature Extractor
     */
    cv::Ptr<cv::Feature2D> features_extractor;

    void init();

    public:
    FeatureExtractor(const ExtractorType::FeatureDescriptorAlgorithm& type) : extractor_type{type} {this->init();}
    ~FeatureExtractor();
    
    Feature* extractFeatures(const cv::Mat& img, const cv::Mat& mask) const;
    
    void updateExtractor(cv::Ptr<cv::Feature2D> new_extractor) {
        this->features_extractor.release();
        this->features_extractor = new_extractor;
    }

    const ExtractorType::FeatureDescriptorAlgorithm& getType() const {return extractor_type;}
    void setType(const ExtractorType::FeatureDescriptorAlgorithm& type) {this->extractor_type = type;}

    

};

#endif
#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <opencv2/opencv.hpp>
#include "../features/Feature.h"

/**
 * @brief Class to represent the type of a extractor.
 */
class ExtractorType{
public:

    /**
     * @brief enum to represent the different types of extractors.
     */
    enum FeatureDescriptorAlgorithm{
        SIFT,
        ORB,
        PHASH,
        COLOR_MOMENT_HASH
    };
   
    static std::string toString(FeatureDescriptorAlgorithm type) {
        switch (type) {
            case SIFT: return "SIFT";
            case ORB: return "ORB";
            case PHASH: return "PHASH";
            case COLOR_MOMENT_HASH: return "COLOR_MOMENT_HASH";
            default: throw std::invalid_argument("Unknown Extractor type");
        }
    }
           
private:

    /**
     * @brief the type of the Extractor
     */
    FeatureDescriptorAlgorithm type;

};

class FeatureExtractor {
public:
    virtual ~FeatureExtractor() = default;

    /**
     * @brief extract features from an image given a mask
     * @param img the image to extract features from
     * @param mask the mask to apply to the image before extracting features (optional)
     * @return a pointer to a Feature object containing the extracted features
     */
    virtual Feature* extractFeatures(const cv::Mat& img, const cv::Mat& mask = cv::Mat()) const = 0;

    /**
     * @brief constructor for the FeatureExtractor class
     * @param type the type of the extractor
     */
    FeatureExtractor(ExtractorType::FeatureDescriptorAlgorithm type) : type_(type) {}
    ExtractorType::FeatureDescriptorAlgorithm getType() const { return type_; }
    void setType(const ExtractorType::FeatureDescriptorAlgorithm& t) { type_ = t; }
protected:

    ExtractorType::FeatureDescriptorAlgorithm type_;
};

#endif // FEATURE_EXTRACTOR_H
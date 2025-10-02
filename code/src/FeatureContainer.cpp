#include "FeatureContainer.h"

const std::map<Card_Type, cv::InputArray>& FeatureDescriptorAlgorithmUtils::get_templates_descriptors(const FeatureDescriptorAlgorithm alg) {
    switch (alg) {
        case FeatureDescriptorAlgorithm::SIFT:
            return FeatureContainer<cv::Mat, FeatureDescriptorAlgorithm::SIFT>::getInstance().get_descriptors();
        case FeatureDescriptorAlgorithm::ORB:
            return FeatureContainer<cv::Mat, FeatureDescriptorAlgorithm::ORB>::getInstance().get_descriptors();
        case FeatureDescriptorAlgorithm::HASH:
            return FeatureContainer<cv::Mat, FeatureDescriptorAlgorithm::HASH>::getInstance().get_descriptors();
        default:
            throw std::invalid_argument("Unsupported FeatureDescriptorAlgorithm");
    }
}
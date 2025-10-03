#include "../../include/features/FeatureContainer.h"

const std::map<Card_Type, Feature*>& FeatureDescriptorAlgorithmUtils::get_templates_features(const FeatureDescriptorAlgorithm alg) {
    switch (alg) {
        case FeatureDescriptorAlgorithm::SIFT:
            return FeatureContainer<FeatureDescriptorAlgorithm::SIFT>::getInstance().get_features();
        case FeatureDescriptorAlgorithm::ORB:
            return FeatureContainer<FeatureDescriptorAlgorithm::ORB>::getInstance().get_features();
        case FeatureDescriptorAlgorithm::HASH:
            return FeatureContainer<FeatureDescriptorAlgorithm::HASH>::getInstance().get_features();
        default:
            throw std::invalid_argument("Unsupported FeatureDescriptorAlgorithm");
    }
}
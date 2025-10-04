#include "../../../../../include/card_detector/objectDetector/featurePipeline/features/FeatureContainer.h"

const std::map<Card_Type, Feature*>& FeatureDescriptorAlgorithmUtils::get_templates_features(const ExtractorType::FeatureDescriptorAlgorithm descriptorAlgorithm) {
    switch (descriptorAlgorithm) {
        case ExtractorType::SIFT:
            return FeatureContainer<ExtractorType::SIFT>::getInstance().get_features();
        case ExtractorType::ORB:
            return FeatureContainer<ExtractorType::ORB>::getInstance().get_features();
        case ExtractorType::HASH:
            return FeatureContainer<ExtractorType::HASH>::getInstance().get_features();
        default:
            throw std::invalid_argument("Unsupported FeatureDescriptorAlgorithm");
    }
}
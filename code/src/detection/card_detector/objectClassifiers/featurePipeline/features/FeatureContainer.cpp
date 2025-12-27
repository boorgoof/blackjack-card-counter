#include "../../../../../../include/detection/card_detector/objectClassifiers/featurePipeline/features/FeatureContainer.h"

const std::map<const ObjectType*, const Feature*>& Utils::FeatureContainerSingleton::get_templates_features(TemplateDataset& dataset, const FeatureExtractor& extractor) {
    switch (extractor.getType()) {
        case ExtractorType::SIFT:
            return FeatureContainer<ExtractorType::SIFT>::getInstance().get_features(dataset, extractor);
        case ExtractorType::ORB:
            return FeatureContainer<ExtractorType::ORB>::getInstance().get_features(dataset, extractor);
        case ExtractorType::HASH:
            return FeatureContainer<ExtractorType::HASH>::getInstance().get_features(dataset, extractor);
        default:
            throw std::invalid_argument("Unsupported FeatureExtractor type");
    }
}
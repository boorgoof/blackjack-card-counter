#include "../../../../../include/card_detector/objectDetector/featurePipeline/features/FeatureContainer.h"

const std::map<ObjectType*, Feature*>& Utils::FeatureContainerSingleton::get_templates_features(const std::string& template_folder_path, const FeatureExtractor& extractor) {
    switch (extractor.getType()) {
        case ExtractorType::SIFT:
            return FeatureContainer<ExtractorType::SIFT>::getInstance().get_features(template_folder_path, extractor);
        case ExtractorType::ORB:
            return FeatureContainer<ExtractorType::ORB>::getInstance().get_features(template_folder_path, extractor);
        case ExtractorType::HASH:
            return FeatureContainer<ExtractorType::HASH>::getInstance().get_features(template_folder_path, extractor);
        default:
            throw std::invalid_argument("Unsupported FeatureExtractor type");
    }
}
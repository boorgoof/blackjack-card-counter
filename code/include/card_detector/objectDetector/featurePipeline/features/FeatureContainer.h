#ifndef FEATURECONTAINER_H
#define FEATURECONTAINER_H

#include <opencv2/opencv.hpp>
#include <map>
#include "CardType.h"
#include "../FeatureExtractor.h"
#include "Feature.h"

//TODO: adapt this class and overall architecture to support different feature extractors other than keypoint-based ones (e.g. image hash)

template<ExtractorType::FeatureDescriptorAlgorithm alg>
class FeatureContainer {
public:
    static FeatureContainer<alg>& getInstance() {
        static FeatureContainer<alg> inst;
        return inst;
    }

    const std::map<Card_Type, Feature*>& get_features(const std::string& template_cards_folder_path, const FeatureExtractor& extractor) const {
        if (!features_) {
            features_ = Loader::TemplateCard::load_template_feature_cards(template_cards_folder_path, extractor);
        }

        return features_;
    }

private:
    FeatureContainer() = default;
    FeatureContainer(const FeatureContainer&) = delete;
    FeatureContainer& operator=(const FeatureContainer&) = delete;

    std::map<Card_Type, Feature*> features_;
};

namespace Utils {
    namespace FeatureContainer {
        const std::map<Card_Type, Feature*>& get_templates_features(const std::string& template_cards_folder_path = nullptr, const FeatureExtractor& extractor = nullptr);
    }
}
#endif // FEATURECONTAINER_H
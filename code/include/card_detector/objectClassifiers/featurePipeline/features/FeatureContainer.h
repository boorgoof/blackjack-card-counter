#ifndef FEATURECONTAINER_H
#define FEATURECONTAINER_H

#include <opencv2/opencv.hpp>
#include <map>
#include "../../../../CardType.h"
#include "../../../../Loaders.h"
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

    const std::map<const ObjectType*, const Feature*>& get_features(TemplateDataset& dataset, const FeatureExtractor& extractor){
        if (!features_) {
            features_ = std::unique_ptr<const std::map<const ObjectType*, const Feature*>>(Loader::TemplateObject::load_template_feature(dataset, extractor));
        }

        return *features_;
    }

private:

    FeatureContainer() = default;
    FeatureContainer(const FeatureContainer&) = delete;
    FeatureContainer& operator=(const FeatureContainer&) = delete;

    std::unique_ptr<const std::map<const ObjectType*, const Feature*>> features_;
};

namespace Utils {
    namespace FeatureContainerSingleton {
        const std::map<const ObjectType*, const Feature*>& get_templates_features(TemplateDataset& dataset, const FeatureExtractor& extractor);
    }
}
#endif // FEATURECONTAINER_H
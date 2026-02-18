//Matteo Bino

#ifndef FEATURECONTAINER_H
#define FEATURECONTAINER_H

#include <opencv2/opencv.hpp>
#include <map>
#include "../../../../../CardType.h"
#include "../../../../../Loaders.h"
#include "../extractors/FeatureExtractor.h"
#include "Feature.h"



class FeatureContainer {
public:
    static FeatureContainer& getInstance() {
        static FeatureContainer inst;
        return inst;
    }

    void loadTemplates(TemplateDataset& dataset, const FeatureExtractor& extractor) {
        features_ = std::unique_ptr<const std::map<const ObjectType*, const Feature*>>(
            Loader::TemplateObject::load_template_feature(dataset, extractor)
        );
    }
    const std::map<const ObjectType*, const Feature*>& getFeatures() const { return *features_; }

private:

    FeatureContainer() = default;
    FeatureContainer(const FeatureContainer&) = delete;
    FeatureContainer& operator=(const FeatureContainer&) = delete;

    std::unique_ptr<const std::map<const ObjectType*, const Feature*>> features_;
};

#endif // FEATURECONTAINER_H
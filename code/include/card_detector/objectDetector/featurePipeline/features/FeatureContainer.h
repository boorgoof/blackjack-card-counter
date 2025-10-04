#ifndef FEATURECONTAINER_H
#define FEATURECONTAINER_H

#include <opencv2/opencv.hpp>
#include <map>
#include "CardType.h"
#include "../FeatureExtractor.h"
#include "Feature.h"


template<ExtractorType::FeatureDescriptorAlgorithm alg>
class FeatureContainer {
public:
    static FeatureContainer<alg>& getInstance() {
        static FeatureContainer<alg> inst;
        return inst;
    }

    void set(std::map<Card_Type, Feature*>&& newMap) {
        features_ = std::move(newMap);
    }

    const std::map<Card_Type, Feature*>& get_features() const {
        return features_;
    }

private:
    FeatureContainer() = default;
    FeatureContainer(const FeatureContainer&) = delete;
    FeatureContainer& operator=(const FeatureContainer&) = delete;

    std::map<Card_Type, Feature*> features_;
};

namespace FeatureDescriptorAlgorithmUtils {
    const std::map<Card_Type, Feature*>& get_templates_features(const ExtractorType::FeatureDescriptorAlgorithm descriptorAlgorithm);
}

#endif // FEATURECONTAINER_H
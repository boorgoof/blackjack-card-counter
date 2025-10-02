#ifndef FEATURECONTAINER_H
#define FEATURECONTAINER_H

#include <opencv2/opencv.hpp>
#include <map>
#include "CardType.h"
#include "FeatureDescriptorAlgorithm.h"



template<typename feature_type, FeatureDescriptorAlgorithm alg>
class FeatureContainer {
public:
    static FeatureContainer<feature_type>& getInstance() {
        static FeatureContainer<feature_type> inst;
        return inst;
    }

    void set(std::map<Card_Type, feature_type>&& newMap) {
        descriptors_ = std::move(newMap);
    }

    const std::map<Card_Type, feature_type>& get_descriptors() const {
        return descriptors_;
    }
    const FeatureDescriptorAlgorithm& get_algorithm() const {
        return algorithm_;
    }

private:
    FeatureContainer() = default;
    FeatureContainer(const FeatureContainer&) = delete;
    FeatureContainer& operator=(const FeatureContainer&) = delete;

    std::map<Card_Type, feature_type> descriptors_;
    FeatureDescriptorAlgorithm algorithm_{alg};
};

namespace FeatureDescriptorAlgorithmUtils {
    const std::map<Card_Type, cv::InputArray>& get_templates_descriptors(const FeatureDescriptorAlgorithm alg);
}

#endif // FEATURECONTAINER_H
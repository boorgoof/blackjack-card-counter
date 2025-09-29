#ifndef FEATURECONTAINER_H
#define FEATURECONTAINER_H

#include <opencv2/opencv.hpp>
#include <map>
#include "CardType.h"



template<typename feature_type>
class FeatureContainer {
public:
    static FeatureContainer<feature_type>& getInstance() {
        static FeatureContainer<feature_type> inst;
        return inst;
    }

    void set(std::map<Card_Type, feature_type>&& newMap) {
        descriptors_ = std::move(newMap);
    }

    const std::map<Card_Type, feature_type>& get() const {
        return descriptors_;
    }

private:
    FeatureContainer() = default;
    FeatureContainer(const FeatureContainer&) = delete;
    FeatureContainer& operator=(const FeatureContainer&) = delete;

    std::map<Card_Type, feature_type> descriptors_;
};

#endif // FEATURECONTAINER_H
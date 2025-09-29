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

// Type-erased interface
class IDescriptorMap {
public:
    virtual ~IDescriptorMap() = default;
    virtual cv::InputArray get(Card_Type type) const = 0;
    virtual std::size_t size() const = 0;
};

// Concrete typed holder
template<typename T>
class DescriptorMapHolder : public IDescriptorMap {
public:
    explicit DescriptorMapHolder(const std::map<Card_Type, T>* map) : map_(map) {}

    cv::InputArray get(Card_Type type) const override {
        auto it = map_->find(type);
        if (it == map_->end()) {
            return cv::noArray();
        }
        return cv::InputArray(it->second);
    }

    std::size_t size() const override { return map_->size(); }

private:
    const std::map<Card_Type, T>* map_;
};

#endif // FEATURECONTAINER_H
#ifndef FEATUREDESCRIPTORALGORITHM_H
#define FEATUREDESCRIPTORALGORITHM_H

#include <opencv2/opencv.hpp>
#include "FeatureContainer.h"

enum class FeatureDescriptorAlgorithm {
    SIFT,
    ORB,
    HASH
};

// Dispatcher 
template<FeatureDescriptorAlgorithm Algo>
struct DescriptorTypeMap;

template<>
struct DescriptorTypeMap<FeatureDescriptorAlgorithm::SIFT> {
    using type = cv::Mat;
};

template<>
struct DescriptorTypeMap<FeatureDescriptorAlgorithm::ORB> {
    using type = cv::Mat;
};

template<>
struct DescriptorTypeMap<FeatureDescriptorAlgorithm::HASH> {
    using type = cv::Mat; // could be std::vector<float> etc.
};

template<FeatureDescriptorAlgorithm Algo>
using FeatureContainerFor = FeatureContainer<typename DescriptorTypeMap<Algo>::type>;

#endif // FEATUREDESCRIPTORALGORITHM_H
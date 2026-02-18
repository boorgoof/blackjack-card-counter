//Matteo Bino

#ifndef HASHFEATURE_H
#define HASHFEATURE_H

#include "Feature.h"
#include <opencv2/opencv.hpp>

/**
 * @brief HashFeature class to represent hash features extracted from images.
 *        This class inherits from the Feature class and contains a cv::Mat to store the hash values.
 */
class HashFeature : public Feature {
    cv::Mat hash_;
public:
    HashFeature(const cv::Mat& h) : hash_(h) {}
    HashFeature(HashFeature&&) = default;
    HashFeature& operator=(HashFeature&&) = default;

    virtual ~HashFeature() = default;

    const cv::Mat& getHash() const { return hash_; }
    void setHash(const cv::Mat& h) { hash_ = h; }
};

#endif // HASHFEATURE_H
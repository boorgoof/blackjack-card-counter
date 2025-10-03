#ifndef KEYPOINTFEATURE_H
#define KEYPOINTFEATURE_H

#include "Feature.h"
#include <opencv2/opencv.hpp>
#include <vector>

class KeypointFeature : public Feature {
    private:
        cv::Mat descriptors_;
        std::vector<cv::KeyPoint> keypoints_;
    public:
        KeypointFeature() = default;
        virtual ~KeypointFeature() = default;

        void setKeypoints(const std::vector<cv::KeyPoint>& keypoints) {
            keypoints_ = keypoints;
        }

        const std::vector<cv::KeyPoint>& getKeypoints() const {
            return keypoints_;
        }

        void setDescriptors(const cv::Mat& descriptors) {
            descriptors_ = descriptors;
        }

        const cv::Mat& getDescriptors() const {
            return descriptors_;
        }
};

#endif // KEYPOINTFEATURE_H
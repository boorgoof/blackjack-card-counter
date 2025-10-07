#ifndef KEYPOINTFEATURE_H
#define KEYPOINTFEATURE_H

#include "Feature.h"
#include <opencv2/opencv.hpp>
#include <vector>

class KeypointFeature : public Feature {
    private:
        cv::Mat descriptors_;
        std::vector<cv::KeyPoint> keypoints_;
        std::vector<cv::Point2f> rect_points_;
    public:

        KeypointFeature() = default;
        KeypointFeature(const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors, const std::vector<cv::Point2f>& rect_points)
            : keypoints_(keypoints), descriptors_(descriptors), rect_points_(rect_points) {}
        KeypointFeature(KeypointFeature&&) = default;
        KeypointFeature& operator=(KeypointFeature&&) = default;
        
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

        void setRectPoints(const std::vector<cv::Point2f>& points) {
            rect_points_ = points;
        }
        const std::vector<cv::Point2f>& getRectPoints() const {
            return rect_points_;
        }
};

#endif // KEYPOINTFEATURE_H
//Federico Meneghetti

#ifndef FEATURES_H
#define FEATURES_H

#include <vector>
#include <opencv2/opencv.hpp>

/**
 * @brief ModelFeatures struct to store the features of a model image
 */
struct ModelFeatures {

    /**
     * @brief index of the model in the dataset
     */
    int dataset_models_idx;

    /**
     * @brief keypoints of the model
     */
    std::vector<cv::KeyPoint> keypoints;

    /**
     * @brief descriptors of the model
     */
    cv::Mat descriptors;

    ModelFeatures(const int models_idx, const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors)
        : dataset_models_idx(models_idx), keypoints(keypoints), descriptors(descriptors) {}
    
    ModelFeatures()
    : dataset_models_idx(-1), keypoints(), descriptors() {}
};


/**
 * @brief SceneFeatures struct to store the features of a scene image
 */
struct SceneFeatures {

    /**
     * @brief keypoints of the scene (test image)
     */
    std::vector<cv::KeyPoint> keypoints;

    /**
     * @brief descriptors of the scene (test image)
     */
    cv::Mat descriptors;

    SceneFeatures( const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors)
        : keypoints(keypoints), descriptors(descriptors) {}

    SceneFeatures():  keypoints(), descriptors() {}
};

#endif
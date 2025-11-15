#ifndef UTILS_H
#define UTILS_H

#include <string>
#include "Label.h"
#include <map>
#include <filesystem>

/**
 * @brief utility functions.
 */
namespace Utils{

    namespace String{
        std::string normalize(const std::string& str); 
    }

    namespace Path{
        std::string longestCommonPath(const std::string& path1_str, const std::string& path2_str);
        std::filesystem::path longestCommonPath(const std::filesystem::path& path1, const std::filesystem::path& path2);
    }

    /**
     * @brief functions for visualizing data and progress.
     */
    namespace Visualization{
        /**
         * @brief Prints a progress bar to the console.
         * @param progress a float between 0 and 1 indicating the progress
         * @param barwidth the width of the progress bar
         * @param prefix a string to print before the progress bar
         * @param suffix a string to print after the progress bar
         */
        void printProgressBar(float progress, size_t barwidth, const std::string& prefix = "", const std::string& suffix = "");
        void showImage(cv::Mat& image, const std::string& window_name = "Image", const int time = 0, const float resize_factor = 1.0);
        void showImage(cv::Mat& image, const std::string& window_name = "Image", const int time = 0, const cv::Size& size = cv::Size());
        /**
         * @brief draws the provided labels directly on the image.
         * @param image the image to draw the labels on
         * @param labels the labels to draw
         * @param box_color the color of the bounding boxes
         * @param text_color the color of the text
         */
        void printLabelsOnImage(cv::Mat& image, const std::vector<Label>& labels, const cv::Scalar& box_color, const cv::Scalar& text_color);
        /**
         * @brief Creates a clone of the input image and shows it with ground truth and predicted labels drawn on it.
         * @param image the image to show
         * @param ground_truth_labels the ground truth labels to draw
         * @param predicted_labels the predicted labels to draw
         * @param window_name the name of the window to show the image in
         */
        void showImageWithLabels(const cv::Mat& image, const cv::Size& size = cv::Size(800, 600), const std::vector<Label>& ground_truth_labels = {}, const std::vector<Label>& predicted_labels = {}, const cv::Scalar& gt_color = cv::Scalar(0,255,0), const cv::Scalar& pred_color = cv::Scalar(255,0,0), const int time=0, const std::string& window_name = "Image with Labels");
        void showImageWithLabels(const cv::Mat& image, const float& resize_factor = 1, const std::vector<Label>& ground_truth_labels = {}, const std::vector<Label>& predicted_labels = {}, const cv::Scalar& gt_color = cv::Scalar(0,255,0), const cv::Scalar& pred_color = cv::Scalar(255,0,0), const int time=0, const std::string& window_name = "Image with Labels");
    }

    /**
     * @brief functions to handle maps.
     */
    namespace Map{
        /**
         * @brief function to create an inverse map from a given map.
         * @tparam MapA2B the type of the map to be inverted
         * @tparam MapB2A the type of the inverted map
         * @param map the map to be inverted
         * @return the inverted map
         * 
         * @note function gently retrieved from //https://stackoverflow.com/questions/54398336/stl-type-for-mapping-one-to-one-relations
         */
        template <typename MapA2B, typename MapB2A = std::map<typename MapA2B::mapped_type, typename MapA2B::key_type>>
        MapB2A createInverseMap(const MapA2B& map){
            MapB2A inverseMap;
            for (const auto& pair : map) {
                inverseMap.emplace(pair.second, pair.first);
            }
            return inverseMap;
        }
    }

};

#endif
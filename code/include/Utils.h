// Matteo Bino

#ifndef UTILS_H
#define UTILS_H

#include <string>
#include "Label.h"
#include "CardType.h"
#include <map>
#include <filesystem>
#include <fstream>
/**
 * @brief utility functions.
 */
namespace Utils{

    namespace String{
        /**
         * @brief function to normalize a string by removing leading and trailing whitespace and converting it to uppercase.
         * @param str the string to normalize
         * @return the normalized string
         */
        std::string normalize(const std::string& str); 
    }

    namespace Path{
        /**
         * @brief function to find the longest common path between two paths.
         * @param path1_str the first path as a string
         * @param path2_str the second path as a string
         * @return the longest common path as a string
         */
        std::string longestCommonPath(const std::string& path1_str, const std::string& path2_str);
        /**
         * @brief function to find the longest common path between two paths.
         * @param path1 the first path as a std::filesystem::path
         * @param path2 the second path as a std::filesystem::path
         * @return the longest common path as a std::filesystem::path
         */
        std::filesystem::path longestCommonPath(const std::filesystem::path& path1, const std::filesystem::path& path2);
    }

    namespace Save{
        /**
         * @brief function to save a vector of labels to a file in YOLO format.
         * @param file_path the path to the file where the labels will be saved
         * @param labels the vector of labels to save
         * @param image_width the width of the image (used to normalize the bounding box coordinates)
         * @param image_height the height of the image (used to normalize the bounding box coordinates)
         * 
         * YOLO format: each line in the file represents a bounding box and has the following format:
         * <class_id> <x_center> <y_center> <width> <height>
         * where:
         * - <class_id> is the integer ID of the class (starting from 0)
         * - <x_center> is the x coordinate of the center of the bounding box, normalized by the width of the image
         * - <y_center> is the y coordinate of the center of the bounding box, normalized by the height of the image
         * - <width> is the width of the bounding box, normalized by the width of the image
         * - <height> is the height of the bounding box, normalized by the height of the image
         */
        void saveLabelsToYoloFile(const std::string& file_path, const std::vector<Label>& labels, const int image_width, const int image_height);
        /**
         * @brief function to save an image to a file.
         * @param file_path the path to the file where the image will be saved
         * @param image the image to save
         */
        void saveImageToFile(const std::string& file_path, const cv::Mat& image);
        /**
         * @brief function to save a confusion matrix to a file.
         * @param file_path the path to the file where the confusion matrix will be saved
         * @param confusion_matrix the confusion matrix to save
         */
        void save_confusion_matrix(const std::string& file_path, const cv::Mat& confusion_matrix);
        /**
         * @brief function to save evaluation metrics to a file.
         * @param file_path the path to the file where the metrics will be saved
         * @param accuracy the accuracy to save
         * @param mean_iou the mean IoU to save
         * @param precision the vector of precision values to save (one for each class)
         * @param recall the vector of recall values to save (one for each class)
         * @param f1 the vector of F1 score values to save (one for each class)
         * @param classes_to_select the set of class indices for which to save the metrics (if empty, all classes will be saved)
         */
        void save_metrics(const std::string& file_path, const float accuracy, const float mean_iou, const std::vector<float>& precision, const std::vector<float>& recall, const std::vector<float>& f1, const std::set<int>& classes_to_select);
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
        /**
         * @brief Shows an image in a window.
         * @param image the image to show
         * @param window_name the name of the window
         * @param time the time to wait before closing the window (0 means wait indefinitely)
         * @param resize_factor the factor by which to resize the image
         */
        void showImage(const cv::Mat& image, const std::string& window_name = "Image", const int time = 0, const float resize_factor = 1.0);
        /**
         * @brief Shows an image in a window.
         * @param image the image to show
         * @param window_name the name of the window
         * @param time the time to wait before closing the window (0 means wait indefinitely)
         * @param size the size to resize the image to
         */
        void showImage(const cv::Mat& image, const std::string& window_name = "Image", const int time = 0, const cv::Size& size = cv::Size());
        /**
         * @brief draws the provided labels directly on the image.
         * @param image the image to draw the labels on
         * @param labels the labels to draw
         * @param box_color the color of the bounding boxes
         * @param text_color the color of the text
         */
        void printLabelsOnImage(cv::Mat& image, const std::vector<Label>& labels, const cv::Scalar& box_color, const cv::Scalar& text_color);
        
        /**
         * @brief draws the provided labels on the image with Hi-Lo color-coded bounding boxes.
         * Green = +1 (2-6), Blue = 0 (7-9), Red = -1 (10, J, Q, K, A)
         * @param image the image to draw the labels on
         * @param labels the labels to draw
         */
        void printLabelsOnImageHiLo(cv::Mat& image, const std::vector<Label>& labels);
      
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
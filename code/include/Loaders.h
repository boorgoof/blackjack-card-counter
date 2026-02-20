// Matteo Bino

#ifndef LOADERS_H
#define LOADERS_H

#include <string>
#include "Label.h"
#include <map>
#include <opencv2/opencv.hpp>
class TemplateDataset;
class FeatureExtractor;
#include "CardType.h"
#include "detection/card_detector/objectClassifiers/featurePipeline/features/Feature.h"

namespace Loader {
    
    namespace Annotation {
        /**
         * @brief load annotations from a yolo format file and return them as a vector of Label objects
         * @param annotation_file_path the path to the yolo annotation file
         * @param image_width the width of the image corresponding to the annotations (used to convert the normalized yolo coordinates to absolute pixel coordinates)
         * @param image_height the height of the image corresponding to the annotations (used to convert the normalized yolo coordinates to absolute pixel coordinates)
         * @return a vector of Label objects representing the annotations in the yolo file
         */
        std::vector<Label> load_yolo_image_annotations(const std::string& annotation_file_path , const int image_width, const int image_height);

        /**
         * @brief convert yolo normalized coordinates to a cv::Rect object representing the bounding box in absolute pixel coordinates
         * @param x_center the x coordinate of the center of the bounding box, normalized to [0,1]
         * @param y_center the y coordinate of the center of the bounding box, normalized to [0,1]
         * @param width the width of the bounding box, normalized to [0,1]
         * @param height the height of the bounding box, normalized to [0,1]
         * @param image_width the width of the image corresponding to the annotations
         * @param image_height the height of the image corresponding to the annotations
         * @return a cv::Rect object representing the bounding box in absolute pixel coordinates
         */
        cv::Rect yoloNorm_to_rect(float x_center, float y_center, float width, float height, int image_width, int image_height);
    };

    namespace Image {
        /**
         * @brief load an image from a file and return it as a cv::Mat object
         * @param image_path the path to the image file
         * @return a cv::Mat object representing the loaded image
         */
        cv::Mat load_image(const std::string& image_path);
    };

    namespace TemplateObject {
        /**
         * @brief load the features of the template objects in the template dataset using the given feature extractor and return them as a map from ObjectType pointers to Feature pointers
         * @param template_dataset the template dataset containing the template objects to load the features from
         * @param extractor the feature extractor to use to extract the features from the template objects
         * @return a map from ObjectType pointers to Feature pointers representing the features of the template objects in the template dataset
         */
        std::map<const ObjectType*, const Feature*>* load_template_feature(TemplateDataset& template_dataset, const FeatureExtractor& extractor);
    }

}

#endif
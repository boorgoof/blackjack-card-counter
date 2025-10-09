#ifndef LOADERS_H
#define LOADERS_H

#include <string>
#include "Label.h"
#include <map>
#include <opencv2/opencv.hpp>
#include "CardType.h"
#include "card_detector/objectDetector/featurePipeline/features/Feature.h"
#include "card_detector/objectDetector/featurePipeline/FeatureExtractor.h"

namespace Loader {
    namespace Annotation {
        std::vector<Label> load_yolo_image_annotations(const std::string& annotation_file_path , const int image_width, const int image_height);
        cv::Rect yoloNorm_to_rect(float x_center, float y_center, float width, float height, int image_width, int image_height);
    
    };

    namespace Image {
        cv::Mat load_image(const std::string& image_path);

    };

    namespace TemplateCard {
        std::map<Card_Type, Feature*>* load_template_feature_cards(const std::string& template_cards_folder_path, const FeatureExtractor& extractor);
    }

}

#endif
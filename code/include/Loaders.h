#ifndef LOADERS_H
#define LOARDERS_H

#include <string>
#include "Label.h"
#include <map>

namespace AnnotationLoaders {

    std::vector<Label> load_yolo_image_annotations(const std::string& annotation_file_path , int image_width, int image_height);
    cv::Rect yoloNorm_to_rect(float x_center, float y_center, float width, float height, int image_width, int image_height);
 
};

#endif
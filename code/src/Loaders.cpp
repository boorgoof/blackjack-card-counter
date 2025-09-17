#include "../include/Loaders.h"
#include <fstream>
#include <opencv2/opencv.hpp>



std::vector<Label> AnnotationLoaders::load_yolo_image_annotations(const std::string& annotation_file_path, int image_width, int image_height){

    std::vector<Label> labels;
    std::ifstream ann_file(annotation_file_path);

    if (!ann_file.is_open()) {
        throw std::runtime_error("Could not open YOLO annotation file: " + annotation_file_path);
        return labels;
    }

    std::string line;
    while (std::getline(ann_file, line)) {

        if (line.empty()) continue;

        std::cout << "Riga : " << line << std::endl;
        
        std::istringstream iss(line);

        int class_id;
        float x_center, y_center, width, height;
        
        if (iss >> class_id >> x_center >> y_center >> width >> height) {

            Card_Type card = Yolo_index_codec::yolo_index_to_card(class_id);
            cv::Rect bbox = yoloNorm_to_rect(x_center, y_center, width, height, image_width, image_height);
            
            labels.emplace_back(card, bbox);

        } else {
            std::cerr << "Error in line: " << line << std::endl;
        }
    }
    
    ann_file.close();
    return labels;


}

cv::Rect AnnotationLoaders::yoloNorm_to_rect(float x_center, float y_center, float width, float height, int image_width, int image_height)
{
    int x = static_cast<int>((x_center - width/2.0f) * image_width);
    int y = static_cast<int>((y_center - height/2.0f) * image_height);
    int w = static_cast<int>(width * image_width);
    int h = static_cast<int>(height * image_height);

    //x = std::max(0, std::min(x, image_width - 1));
    //y = std::max(0, std::max(y, image_height - 1));
    //w = std::max(1, std::min(w, image_width - x));
    //h = std::max(1, std::max(h, image_height - y));

    
    return cv::Rect(x, y, w, h);
}

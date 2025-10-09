#include "../include/Loaders.h"

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <regex>


std::vector<Label> Loader::Annotation::load_yolo_image_annotations(const std::string& annotation_file_path, const int image_width, const int image_height){

    std::vector<Label> labels;
    std::ifstream ann_file(annotation_file_path);

    if (!ann_file.is_open()) {
        throw std::runtime_error("Could not open YOLO annotation file: " + annotation_file_path);
        return labels;
    }

    std::string line;
    while (std::getline(ann_file, line)) {

        if (line.empty()) continue;
        
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

cv::Rect Loader::Annotation::yoloNorm_to_rect(float x_center, float y_center, float width, float height, int image_width, int image_height)
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

cv::Mat Loader::Image::load_image(const std::string &image_path)
{
    //check if the file exists
    if (!std::filesystem::exists(image_path)) {
        throw std::runtime_error("Image file not found: " + image_path);
    }
    return cv::imread(image_path, cv::IMREAD_COLOR);
}

std::map<Card_Type, Feature*>* Loader::TemplateCard::load_template_feature_cards(const std::string &template_cards_folder_path, const FeatureExtractor& extractor)
{
    
    //check if the folder exists
    if (!std::filesystem::exists(template_cards_folder_path)) {
        throw std::runtime_error("Template cards folder not found: " + template_cards_folder_path);
    }
    
    //for each filename in the folder, load the image and compute the feature descriptor
    std::map<Card_Type, Feature*>* template_feature_cards = new std::map<Card_Type, Feature*>();
    for (const auto & entry : std::filesystem::directory_iterator(template_cards_folder_path)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();
            std::string file_name = entry.path().filename().string();

            //extract the card type from the filename (name is [letter][number].png, e.g. CA.png for Ace of Clubs, HK.png for King of Hearts, etc.)
            //regex to match the pattern where the first character is one of C, D, H, S
            //and the second part is either the substring "10" or a single character among A,2,3,4,5,6,7,8,9,T,J,Q,K
            std::regex card_regex("([CDHS])((10)|[A23456789TJQK])\\.png");
            std::smatch match;
            if (std::regex_search(file_name, match, card_regex)) {
                Card_Type card_type = Card_Type(Card_Type::string_to_rank(match[2].str()), Card_Type::string_to_suit(match[1].str()));
                if (!card_type.isValid()) {
                    std::cerr << "Unknown card type in template card filename: " << file_name << std::endl;
                    continue;
                }
                (*template_feature_cards)[card_type] = extractor.extractFeatures(Loader::Image::load_image(file_path), cv::Mat());
            } else {
                std::cerr << "Invalid template card filename: " << file_name << std::endl;
                continue;
            }

            
        }
    }

    return template_feature_cards;
}
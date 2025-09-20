#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include "../include/Label.h"
#include "../include/CardType.h"
#include "../include/Utils.h"
#include "../include/Loaders.h"
#include "../include/ImageFilter.h"
#include "../include/CardDetector.h"
#include "../include/SequentialCardDetector.h"
#include "../include/SingleCardDetector.h"
#include "../include/Dataset.h"

int main(int argc, char** argv) {

    if (argc != 3) {
        std::cerr << "Usage: ./program <datasets_path> <output_path>" << std::endl;
        return 1;
    }
    std::string datasets_path = argv[1];
    std::string output_path = argv[2];

    if (!std::filesystem::exists(datasets_path)) {
        std::cerr << "The datasets path does not exist!" << std::endl;
        return 1;
    }

    if (std::filesystem::exists(output_path)) {
        std::cout << "The output path already exists! Do you want to proceed? (y/n)";
        char response;
        std::cin >> response;
        if (response != 'y' && response != 'Y') {
            std::cout << "Exiting the program." << std::endl;
            return 0;
        }
        else{
            std::cout << "Overwriting the output path!" << std::endl;
            std::filesystem::remove_all(output_path);
            std::filesystem::create_directories(output_path);
        }
    } else {
        std::filesystem::create_directories(output_path);
        std::cout << "The output path has been created!" << std::endl;
    }

    std::cout << "datasets_path: " << datasets_path << std::endl;
    std::cout << "output_path: " << output_path << std::endl;


    std::string single_cards_dataset_path = datasets_path + "/single_cards";
    std::string videos_dataset_path = datasets_path + "/videos";

    //Dataset object creation
    Dataset single_cards_dataset(single_cards_dataset_path, false); //false is the value for flag "is_sequential"
    Dataset::Iterator it = single_cards_dataset.begin();

    //depending on the dataset type, create the appropriate card detector (specific parameters will be decided later, in the actual implementation)
    std::unique_ptr<CardDetector> card_detector = nullptr;
    bool detect_full_card = false; //depending on the dataset, we may want to detect the full card or just a part of it (e.g., the rank and suit in the corner)
    if (single_cards_dataset.is_sequential()) {
        card_detector = std::make_unique<SequentialCardDetector>(detect_full_card);
    } else {
        card_detector = std::make_unique<SingleCardDetector>(detect_full_card);
    }

    ImageFilter img_filter;
    img_filter.add_filter("Resize", Filters::resize, 0.5, 0.5); //resize to halve image size in both dimensions, 1/4 computational cost (check if performances decrease or not)
    img_filter.add_filter("Gaussian Blur", Filters::gaussian_blur, cv::Size(7,7)); //check if it is useful for robustness to noise or just useless

    //prepare a vector to store the predicted labels for every image
    std::vector<std::pair<std::string, std::vector<Label>>> single_cards_labels = std::vector<std::pair<std::string, std::vector<Label>>>();

    for ( ; it != single_cards_dataset.end(); ++it) {
        ImageInfo img_info = *it;
        //here you should see the image filename, image path, annotation path, is_annotated flag
        cv::Mat img = Loaders::load_image(img_info.image_path);
        if (img.empty()) {
            std::cerr << "Could not read the image: " << img_info.image_path << std::endl;
            continue;
        }
        //apply preprocessing filters
        img = img_filter.apply_filters(img);
        //adds the result of the detection to the vector
        single_cards_labels.push_back({img_info.image_filename, card_detector->detect_image(img)});
    }

    //calculate metrics for single cards dataset


    //same for sequential cards dataset


    //cv::Mat img(200, 300, CV_8UC3, cv::Scalar(40, 40, 200));
    //cv::putText(img, "Hello, OpenCV!", {20, 110}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {255,255,255}, 2);
    //cv::imshow("win", img);
    //cv::waitKey(0);

    Card_Type card("10S");
    Card_Type card2("10000S");  
    Card_Type card3("AS");  
    Card_Type card4("ASP");
    Card_Type card5("10C");  
    std::cout << card << std::endl;
    std::cout << card2 << std::endl;
    std::cout << card3 << std::endl;
    
    
   
    
    cv::Mat img;
    img = cv::imread("../dataset/images/image1.png");
    if (img.empty()) {
        std::cerr << "Could not read the image: " << std::endl;
        return 1;
    }

    std::vector<Label> labels = AnnotationLoaders::load_yolo_image_annotations("../datasetTest/labels/image1.txt", img.cols, img.rows);
    for(int i = 0; i < labels.size(); i++){
        std::cout << labels[i] << std::endl;

        const cv::Rect& rect = labels[i].get_bounding_box();
        cv::rectangle(img, rect, cv::Scalar(0,255,0), 2, cv::LINE_AA);

    }

    cv::imshow("Display window", img);
    cv::waitKey(0);



    



    return 0;
}
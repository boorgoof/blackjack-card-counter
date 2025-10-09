// main.cpp
#include <iostream>
#include <filesystem>
#include "../include/Label.h"
#include "../include/Utils.h"
#include "../include/Loaders.h"
#include "../include/ImageFilter.h"
#include "../include/card_detector/CardDetector.h"
#include "../include/card_detector/SequentialCardDetector.h"
#include "../include/card_detector/SingleCardDetector.h"
#include "../include/Dataset.h"
#include "../include/StatisticsCalculation.h"
#include "../include/card_detector/objectDetector/featurePipeline/features/FeatureContainer.h"
#include "../include/card_detector/objectDetector/featurePipeline/FeaturePipeline.h"

int main(int argc, char** argv) {
    //TODO: use a proper argument parser library or make this more flexible
    if (argc < 3) {
        std::cerr << "Usage: ./program <datasets_path> <output_path> <visualize_flag>" << std::endl;
        std::cerr << "datasets_path: path to the folder containing the datasets (single_cards and videos folders)" << std::endl;
        std::cerr << "output_path: path to the folder where the output will be saved" << std::endl;
        std::cerr << "visualize_flag: FOR NOW JUST FOR DEVELOPMENT PURPOSE whether to visualize the detected images (true/false), optional, default is false" << std::endl;
        return 1;
    }
    std::string datasets_path = argv[1];
    std::string output_path = argv[2];
    bool visualize = (argc > 3) ? (std::string(argv[3]) == "true") : false;

    if (!std::filesystem::exists(datasets_path)) {
        std::cerr << "The datasets path does not exist!" << std::endl;
        return 1;
    }

    if (std::filesystem::exists(output_path)) {
        std::cout << "The output path already exists! Do you want to proceed? (y/n): ";
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

    std::cout << "Program started with the following parameters:" << std::endl;
    std::cout << "datasets_path: " << datasets_path << std::endl;
    std::cout << "output_path: " << output_path << std::endl;
    std::cout << "visualize: " << (visualize ? "true" : "false") << std::endl;

    std::string single_cards_dataset_path = datasets_path + "/single_cards";
    std::string videos_dataset_path = datasets_path + "/videos";

    //Dataset object creation
    Dataset single_cards_dataset(single_cards_dataset_path, false);
    Dataset::Iterator it = single_cards_dataset.begin();

    const std::string template_cards_folder_path = "../data/template/crop_template";

    //depending on the dataset type, create the appropriate card detector (specific parameters will be decided later, in the actual implementation)
    std::unique_ptr<CardDetector> card_detector = nullptr;
    bool detect_full_card = false; //depending on the dataset, we may want to detect the full card or just a part of it (e.g., the rank and suit in the corner)
    if (single_cards_dataset.is_sequential()) {
        card_detector = std::make_unique<SequentialCardDetector>(detect_full_card, visualize);
    } else {
        card_detector = std::make_unique<SingleCardDetector>(RoughCardDetector(), new FeaturePipeline(ExtractorType::SIFT, MatcherType::FLANN), detect_full_card, visualize);
    }

    ImageFilter img_filter;
    img_filter.add_filter("Resize", Filters::resize, 0.5, 0.5); //resize to halve image size in both dimensions, 1/4 computational cost (check if performances decrease or not)
    //img_filter.add_filter("Gaussian Blur", Filters::gaussian_blur, cv::Size(7,7)); //check if it is useful for robustness to noise or just useless

    //prepare a vector to store the predicted labels and the ground truth for every image
    std::vector<std::vector<Label>> predicted_labels = std::vector<std::vector<Label>>();
    std::vector<std::vector<Label>> true_labels = std::vector<std::vector<Label>>();

    //keep track of the time taken to load and detect each image
    std::chrono::duration<double, std::milli> total_loading_time;
    std::chrono::duration<double, std::milli> total_detection_time;
    for ( ; it != single_cards_dataset.end(); ++it) {
        auto start = std::chrono::steady_clock::now();

        ImageInfo img_info = *it;

        cv::Mat img = Loader::Image::load_image(img_info.get_pathImage());

        img = img_filter.apply_filters(img);

        auto loading_time = std::chrono::steady_clock::now();
        total_loading_time += std::chrono::duration<double, std::milli>(loading_time - start);

        //detects card in image and adds the result of the detection to the vector
        predicted_labels.push_back(card_detector->detect_image(img));

        auto detection_time = std::chrono::steady_clock::now();
        total_detection_time += std::chrono::duration<double, std::milli>(detection_time - loading_time);

        true_labels.push_back(Loader::Annotation::load_yolo_image_annotations(img_info.get_pathLabel(), img.cols, img.rows));

        Utils::Visualization::printProgressBar(static_cast<float>(std::distance(single_cards_dataset.begin(), it) + 1) / std::distance(single_cards_dataset.begin(), single_cards_dataset.end()),
                                                 50, "Processing images: ", "Complete");
    }

    std::cout << "Dataset: " << single_cards_dataset.get_path() << std::endl;
    std::cout << "Total images processed: " << std::distance(single_cards_dataset.begin(), single_cards_dataset.end()) << std::endl;
    std::cout << "Average loading time per image: " << total_loading_time.count() / std::distance(single_cards_dataset.begin(), single_cards_dataset.end()) << " ms" << std::endl;
    std::cout << "Average detection time per image: " << total_detection_time.count() / std::distance(single_cards_dataset.begin(), single_cards_dataset.end()) << " ms" << std::endl;

    /**
    //calculate metrics for single cards dataset
    cv::Mat cfm = StatisticsCalculation::calc_confusion_matrix(true_labels, predicted_labels, 52); //52 classes for a standard deck of cards
    StatisticsCalculation::print_confusion_matrix(cfm);

    StatisticsCalculation::calc_dataset_meanIoU(true_labels, predicted_labels);
    StatisticsCalculation::print_dataset_meanIoU();
    */


    //TODO: repeat the same process for the videos
}

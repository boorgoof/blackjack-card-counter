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
#include "../include/card_detector/RoughCardDetector.h"
#include "../include/Dataset/ImageDataset.h"
#include "../include/Dataset/VideoDataset.h"
#include "../include/Dataset/TemplateDataset.h"
#include "../include/SampleInfo/TemplateInfo.h"
#include "../include/StatisticsCalculation.h"
#include "../include/card_detector/objectClassifiers/featurePipeline/features/FeatureContainer.h"
#include "../include/card_detector/objectClassifiers/featurePipeline/FeaturePipeline.h"
#include "../include/card_detector/objectSegmenters/SimpleContoursCardSegmenter.h"
#include "../include/card_detector/objectSegmenters/DistanceTransformCardSegmenter.h"

int main(int argc, char** argv) {
    //TODO: use a proper argument parser library or make this more flexible
    if (argc < 4) {
        std::cerr << "Usage: ./program <datasets_path> <template_dataset_path> <output_path> <visualize_flag>" << std::endl;
        std::cerr << "datasets_path: path to the folder containing the datasets (single_cards and videos folders)" << std::endl;
        std::cerr << "template_dataset_path: path to the folder containing the template cards dataset" << std::endl;
        std::cerr << "output_path: path to the folder where the output will be saved" << std::endl;

        std::cerr << "visualize_flag: FOR NOW JUST FOR DEVELOPMENT PURPOSE whether to visualize the detected images (true/false), optional, default is false" << std::endl;
        return 1;
    }
    std::string datasets_path = argv[1];
    std::string template_dataset_path = argv[2];
    std::string output_path = argv[3];
    bool visualize = (argc > 4) ? (std::string(argv[4]) == "true") : false;

    if (!std::filesystem::exists(datasets_path)) {
        std::cerr << "The datasets path does not exist!" << std::endl;
        return 1;
    }

    if (!std::filesystem::exists(template_dataset_path)) {
        std::cerr << "The template dataset path does not exist!" << std::endl;
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
    std::cout << "template_dataset_path: " << template_dataset_path << std::endl;
    std::cout << "output_path: " << output_path << std::endl;
    std::cout << "visualize: " << (visualize ? "true" : "false") << std::endl;

    std::string single_cards_dataset_path = datasets_path + "/single_cards";
    std::string videos_dataset_path = datasets_path + "/videos";

    //TemplateDataset creation
    TemplateDataset template_dataset(template_dataset_path);
    std::cout << "Template Dataset root: " << template_dataset.get_root() << std::endl;
    std::cout << "Template Dataset loaded with " << template_dataset.size() << " entries." << std::endl;
    for( auto it = template_dataset.begin(); it != template_dataset.end(); ++it) {
        const TemplateInfo& sample = dynamic_cast<const TemplateInfo&>(*it);
        std::cout << "Template card: " << sample.get_name() << ", Path: " << sample.get_pathSample() << std::endl;
        if (visualize){
            cv::Mat img = template_dataset.load(it);
            cv::imshow("Template Card: " + sample.get_name(), img);
            cv::waitKey(200);
            cv::destroyAllWindows();
        }
    }

    //Dataset object creation
    ImageDataset single_cards_dataset(single_cards_dataset_path);

    //depending on the dataset type, create the appropriate card detector (specific parameters will be decided later, in the actual implementation)
    std::unique_ptr<CardDetector> card_detector = nullptr;
    bool detect_full_card = false; //depending on the dataset, we may want to detect the full card or just a part of it (e.g., the rank and suit in the corner)
    if (single_cards_dataset.is_sequential()) {
        card_detector = std::make_unique<SequentialCardDetector>(detect_full_card, visualize);
    } else {
        card_detector = std::make_unique<SingleCardDetector>(new RoughCardDetector(PipelinePreset::DEFAULT, MaskType::POLYGON), new FeaturePipeline(ExtractorType::FeatureDescriptorAlgorithm::SIFT, MatcherType::MatcherAlgorithm::FLANN, template_dataset), new  SimpleContoursCardSegmenter(),  detect_full_card, visualize);
    }

    ImageFilter img_filter;
    img_filter.add_filter("Resize", Filters::resize, 0.5, 0.5); //resize to halve image size in both dimensions, 1/4 computational cost (check if performances decrease or not)
    
    //prepare a vector to store the predicted labels and the ground truth for every image
    std::vector<std::vector<Label>> predicted_labels = std::vector<std::vector<Label>>();
    std::vector<std::vector<Label>> true_labels = std::vector<std::vector<Label>>();

    //keep track of the time taken to load and detect each image
    std::chrono::duration<double, std::milli> total_loading_time;
    std::chrono::duration<double, std::milli> total_detection_time;
    for (auto it = single_cards_dataset.begin(); it != single_cards_dataset.end(); ++it) {
        auto start = std::chrono::steady_clock::now();

        SampleInfo* img_info = &(*it);

        cv::Mat img = single_cards_dataset.load(it);

        img = img_filter.apply_filters(img);

        auto loading_time = std::chrono::steady_clock::now();
        total_loading_time += std::chrono::duration<double, std::milli>(loading_time - start);

        //detects card in image and adds the result of the detection to the vector
        predicted_labels.push_back(card_detector->detect_image(img));

        auto detection_time = std::chrono::steady_clock::now();
        total_detection_time += std::chrono::duration<double, std::milli>(detection_time - loading_time);

        true_labels.push_back(Loader::Annotation::load_yolo_image_annotations(img_info->get_pathLabel(), img.cols, img.rows));

        if(visualize){
            cv::Mat vis_img = img.clone();
            //draw true labels in green
            for (const auto& label : true_labels.back()) {
                cv::rectangle(vis_img, label.get_bounding_box(), cv::Scalar(0, 255, 0), 2);
                if (label.get_object()) {
                    cv::putText(vis_img, label.get_object()->to_string(), cv::Point(label.get_bounding_box().x, label.get_bounding_box().y - 10),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                }
            }
            //draw predicted labels in red
            for (const auto& label : predicted_labels.back()) {
                cv::rectangle(vis_img, label.get_bounding_box(), cv::Scalar(0, 0, 255), 2);
                if (label.get_object()) {
                    cv::putText(vis_img, label.get_object()->to_string(), cv::Point(label.get_bounding_box().x, label.get_bounding_box().y - 10),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
                }
            }
            cv::imshow("Detections", vis_img);
            cv::waitKey(500); //display each image for 500 ms
            cv::destroyAllWindows();
        }

        Utils::Visualization::printProgressBar(static_cast<float>(std::distance(single_cards_dataset.begin(), it) + 1) / std::distance(single_cards_dataset.begin(), single_cards_dataset.end()),
                                                 50, "Processing images: ", "Complete");
    }

    std::cout << "Dataset image path: " << single_cards_dataset.get_root() << std::endl;
    std::cout << "Dataset annotation path: " << single_cards_dataset.get_annotation_root() << std::endl;
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

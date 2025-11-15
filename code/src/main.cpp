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

std::unique_ptr<CardDetector> create_card_detector_for_dataset(const std::unique_ptr<Dataset>& dataset, TemplateDataset& template_dataset, const bool detect_full_card, const bool visualize);
void iterate_dataset(std::unique_ptr<Dataset>& dataset, const ImageFilter& image_filter, std::unique_ptr<CardDetector>& card_detector, const std::string& output_folder_path, const bool visualize);

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

    std::string single_cards_folder = "single_cards";
    std::string videos_folder = "videos";

    std::string single_cards_dataset_path = datasets_path + "/" + single_cards_folder;
    std::string videos_dataset_path = datasets_path + "/" + videos_folder;

    //TemplateDataset creation
    TemplateDataset template_dataset(template_dataset_path);
    std::cout << "Template Dataset root: " << template_dataset.get_root() << std::endl;
    std::cout << "Template Dataset loaded with " << template_dataset.size() << " entries." << std::endl;
    if (visualize) {
        for (auto it = template_dataset.begin(); it != template_dataset.end(); ++it) {
            const TemplateInfo& sample = dynamic_cast<const TemplateInfo&>(*it);
            cv::Mat img = template_dataset.load(it);
            Utils::Visualization::showImage(img, "Template Card: " + sample.get_name(), 200, 1);
        }
    }

    //----------  SINGLE CARD DATASET  ----------

    //Dataset object creation
    std::unique_ptr<Dataset> single_cards_dataset(new ImageDataset(single_cards_dataset_path));

    //depending on the dataset type, create the appropriate card detector
    // change create_card_detector_for_dataset to return std::unique_ptr<CardDetector>
    std::unique_ptr<CardDetector> card_detector = create_card_detector_for_dataset(single_cards_dataset, template_dataset, false, visualize);

    ImageFilter img_filter;
    img_filter.add_filter("Resize", Filters::resize, 0.5, 0.5); //resize to halve image size in both dimensions, 1/4 computational cost (check if performances decrease or not)

    iterate_dataset(single_cards_dataset, img_filter, card_detector, output_path + "/" + single_cards_folder, visualize);

    //----------  VIDEO DATASET  ----------

    //Dataset object creation
    std::unique_ptr<Dataset> video_dataset(new VideoDataset(videos_dataset_path));

    //depending on the dataset type, create the appropriate card detector
    card_detector.release();
    card_detector = create_card_detector_for_dataset(video_dataset, template_dataset, true, visualize);

    iterate_dataset(video_dataset, img_filter, card_detector, output_path + "/" + videos_folder, visualize);
}

std::unique_ptr<CardDetector> create_card_detector_for_dataset(const std::unique_ptr<Dataset>& dataset, TemplateDataset& template_dataset, const bool detect_full_card, const bool visualize) {
    if (dataset->is_sequential()) {
        return std::make_unique<SequentialCardDetector>(detect_full_card, visualize);
    } else {
        return std::make_unique<SingleCardDetector>(std::make_unique<RoughCardDetector>(PipelinePreset::DEFAULT, MaskType::POLYGON), std::make_unique<FeaturePipeline>(ExtractorType::FeatureDescriptorAlgorithm::SIFT, MatcherType::MatcherAlgorithm::FLANN, template_dataset), std::make_unique<SimpleContoursCardSegmenter>(), detect_full_card, visualize);
    }
}

void iterate_dataset(std::unique_ptr<Dataset>& dataset, const ImageFilter& image_filter, std::unique_ptr<CardDetector>& card_detector, const std::string& output_folder_path, const bool visualize){

    std::string annotations_folder = output_folder_path + "/annotations/";
    std::string images_folder = output_folder_path + "/images/";

    if (!std::filesystem::exists(annotations_folder)) {
        std::filesystem::create_directories(annotations_folder);
    }
    if (!std::filesystem::exists(images_folder)) {
        std::filesystem::create_directories(images_folder);
    }

    //keep track of the time taken to load and detect each image
    std::chrono::duration<double, std::milli> total_loading_time;
    std::chrono::duration<double, std::milli> total_detection_time;
    for (auto it = dataset->begin(); it != dataset->end(); ++it) {
        //vectors to hold predicted and true labels for the current image
        std::vector<Label> predicted_labels;
        std::vector<Label> true_labels;

        auto start = std::chrono::steady_clock::now();

        SampleInfo* img_info = &(*it);

        cv::Mat img = dataset->load(it);

        img = image_filter.apply_filters(img);

        auto loading_time = std::chrono::steady_clock::now();
        total_loading_time += std::chrono::duration<double, std::milli>(loading_time - start);

        //detects cards in image and adds the result of the detection to the vector
        predicted_labels = card_detector->detect_image(img);

        auto detection_time = std::chrono::steady_clock::now();
        total_detection_time += std::chrono::duration<double, std::milli>(detection_time - loading_time);

        true_labels = Loader::Annotation::load_yolo_image_annotations(img_info->get_pathLabel(), img.cols, img.rows);

        //saves the predicted labels to a file
        Utils::Save::saveLabelsToYoloFile(annotations_folder + img_info->get_name() + ".txt", predicted_labels, img.cols, img.rows);

        cv::Mat output_img = img.clone();
        Utils::Visualization::printLabelsOnImage(output_img, true_labels, cv::Scalar(0,255,0), cv::Scalar(0,255,0)); //true labels in green
        Utils::Visualization::printLabelsOnImage(output_img, predicted_labels, cv::Scalar(255,0,0), cv::Scalar(255,0,0)); //predicted labels in red

        Utils::Save::saveImageToFile(images_folder + img_info->get_name() + ".png", output_img);

        if(visualize){
            Utils::Visualization::showImage(output_img, img_info->get_name(), 1000, 0.5);
        }

        Utils::Visualization::printProgressBar(static_cast<float>(std::distance(dataset->begin(), it) + 1) / std::distance(dataset->begin(), dataset->end()),
                                                 50, "Processing images: ", "Complete");
    }

    std::cout << "Dataset image path: " << dataset->get_root() << std::endl;
    std::cout << "Dataset annotation path: " << dataset->get_annotation_root() << std::endl;
    std::cout << "Total images processed: " << std::distance(dataset->begin(), dataset->end()) << std::endl;
    std::cout << "Average loading time per image: " << total_loading_time.count() / std::distance(dataset->begin(), dataset->end()) << " ms" << std::endl;
    std::cout << "Average detection time per image: " << total_detection_time.count() / std::distance(dataset->begin(), dataset->end()) << " ms" << std::endl;

    /**
    //calculate metrics for single cards dataset
    cv::Mat cfm = StatisticsCalculation::calc_confusion_matrix(true_labels, predicted_labels, 52); //52 classes for a standard deck of cards
    StatisticsCalculation::print_confusion_matrix(cfm);

    StatisticsCalculation::calc_dataset_meanIoU(true_labels, predicted_labels);
    StatisticsCalculation::print_dataset_meanIoU();
    */

}
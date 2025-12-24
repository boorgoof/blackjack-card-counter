// main.cpp
#include <iostream>
#include <filesystem>
#include "../include/Label.h"
#include "../include/Utils.h"
#include "../include/Loaders.h"
#include "../include/ImageFilter.h"
#include "../include/detection/ProcessingMode.h"
#include "../include/detection/SequentialFrameProcessing.h"
#include "../include/detection/SingleFrameProcessing.h"
#include "../include/detection/card_detector/MaskCardDetector.h"
#include "../include/detection/card_detector/SegmentationClassificationCardDetector.h"
#include "../include/Dataset/ImageDataset.h"
#include "../include/Dataset/VideoDataset.h"
#include "../include/Dataset/TemplateDataset.h"
#include "../include/SampleInfo/TemplateInfo.h"
#include "../include/StatisticsCalculation.h"
#include "../include/detection/card_detector/objectClassifiers/featurePipeline/features/FeatureContainer.h"
#include "../include/detection/card_detector/objectClassifiers/featurePipeline/FeaturePipeline.h"
#include "../include/detection/card_detector/objectSegmenters/SimpleContoursCardSegmenter.h"
#include "../include/detection/card_detector/objectSegmenters/DistanceTransformCardSegmenter.h"

std::unique_ptr<ProcessingMode> create_mode_for_dataset(const std::unique_ptr<Dataset>& dataset, TemplateDataset& template_dataset, const bool detect_full_card, const bool visualize);
void iterate_dataset(std::unique_ptr<Dataset>& dataset, const ImageFilter& image_filter, std::unique_ptr<ProcessingMode>& mode, const std::string& output_folder_path, const bool visualize, const int num_classes = 53, const float iou_threshold = 0.5f);

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

    constexpr int num_classes = 53; //52 cards + background/no card class
    constexpr float iou_threshold = 0.5f;

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
    // change create_mode_for_dataset to return std::unique_ptr<ProcessingMode>
    std::unique_ptr<ProcessingMode> mode = create_mode_for_dataset(single_cards_dataset, template_dataset, false, visualize);

    ImageFilter img_filter;
    img_filter.add_filter("Resize", Filters::resize, 0.5, 0.5); //resize to halve image size in both dimensions, 1/4 computational cost (check if performances decrease or not)

    iterate_dataset(single_cards_dataset, img_filter, mode, output_path + "/" + single_cards_folder, visualize, num_classes);

    //----------  VIDEO DATASET  ----------

    //Dataset object creation
    std::unique_ptr<Dataset> video_dataset(new VideoDataset(videos_dataset_path));

    //depending on the dataset type, create the appropriate card detector
    mode.release();
    mode = create_mode_for_dataset(video_dataset, template_dataset, true, visualize);

    iterate_dataset(video_dataset, img_filter, mode, output_path + "/" + videos_folder, visualize, num_classes);
}

// todo onestamente io toglierei il complete card. Facciamo gli angolini e stop ho cambiato la funzione rispetto a prima e tengo solo gli angolini
std::unique_ptr<ProcessingMode> create_mode_for_dataset(const std::unique_ptr<Dataset>& dataset, TemplateDataset& template_dataset, const bool detect_full_card, const bool visualize) {
    if (dataset->is_sequential()) {
        return std::make_unique<SequentialFrameProcessing>(detect_full_card, visualize);
    } else {
        auto card_detector = std::make_unique<SegmentationClassificationCardDetector>(std::make_unique<MaskCardDetector>(PipelinePreset::DEFAULT, MaskType::POLYGON), std::make_unique<FeaturePipeline>( ExtractorType::FeatureDescriptorAlgorithm::SIFT, MatcherType::MatcherAlgorithm::FLANN,template_dataset),std::make_unique<SimpleContoursCardSegmenter>());
        // Single-frame processing that owns the detector
        return std::make_unique<SingleFrameProcessing>(std::move(card_detector));
    }
}

void iterate_dataset(std::unique_ptr<Dataset>& dataset, const ImageFilter& image_filter, std::unique_ptr<ProcessingMode>& mode, const std::string& output_folder_path, const bool visualize, const int num_classes, const float iou_threshold) {

    std::string annotations_folder = output_folder_path + "/annotations/";
    std::string images_folder = output_folder_path + "/images/";
    std::string stats_folder = output_folder_path + "/stats/";

    
    if (!std::filesystem::exists(annotations_folder)) {
        std::filesystem::create_directories(annotations_folder);
    }
    if (!std::filesystem::exists(images_folder)) {
        std::filesystem::create_directories(images_folder);
    }
    if (!std::filesystem::exists(stats_folder)) {
        std::filesystem::create_directories(stats_folder);
    }

    //keep track of the time taken to load and detect each image
    std::chrono::duration<double, std::milli> total_load_time{0};
    std::chrono::duration<double, std::milli> total_detect_time{0};
    std::chrono::duration<double, std::milli> total_gt_time{0};
    std::chrono::duration<double, std::milli> total_save_annotations_time{0};
    std::chrono::duration<double, std::milli> total_draw_time{0};
    std::chrono::duration<double, std::milli> total_save_image_time{0};
    std::chrono::duration<double, std::milli> total_total_time{0};

    const auto total_images = std::distance(dataset->begin(), dataset->end());
    int idx = 0;

    cv::Mat cumulative_confusion_matrix = cv::Mat::zeros(num_classes, num_classes, CV_32S);
    const int save_cm_every = 10;

    for (auto it = dataset->begin(); it != dataset->end(); ++it) {

        //vectors to hold predicted and true labels for the current image
        std::vector<Label> predicted_labels;
        std::vector<Label> true_labels;

        auto time_start = std::chrono::steady_clock::now();

        //load and filter image
        SampleInfo* img_info = &(*it);
        cv::Mat img = dataset->load(it);
        img = image_filter.apply_filters(img);
        auto time_load_end = std::chrono::steady_clock::now();
        double load_ms = std::chrono::duration<double, std::milli>(time_load_end - time_start).count();

        //detects cards in image and adds the result of the detection to the vector
        predicted_labels = mode->detect_image(img);
        auto time_detect_end = std::chrono::steady_clock::now();
        double detect_ms = std::chrono::duration<double, std::milli>(time_detect_end - time_load_end).count();

        //load ground truth labels
        true_labels = Loader::Annotation::load_yolo_image_annotations(img_info->get_pathLabel(), img.cols, img.rows);
        auto time_gt_end = std::chrono::steady_clock::now();
        double gt_ms = std::chrono::duration<double, std::milli>(time_gt_end - time_detect_end).count();

        //saves the predicted labels to a file
        Utils::Save::saveLabelsToYoloFile(annotations_folder + img_info->get_name() + ".txt", predicted_labels, img.cols, img.rows);
        auto time_save_pred_ann_end = std::chrono::steady_clock::now();
        double save_pred_ann_ms = std::chrono::duration<double, std::milli>(time_save_pred_ann_end - time_gt_end).count();

        //update cumulative confusion matrix
        cumulative_confusion_matrix += StatisticsCalculation::calc_confusion_matrix(true_labels, predicted_labels, num_classes, iou_threshold);
        ++idx;
        float accuracy = StatisticsCalculation::calc_accuracy(cumulative_confusion_matrix);
        if (save_cm_every > 0 && (idx % save_cm_every == 0)) {
            Utils::Save::save_confusion_matrix(stats_folder + "confusion_matrix.txt", cumulative_confusion_matrix);

            std::vector<float> precision = StatisticsCalculation::calc_precision(cumulative_confusion_matrix);
            std::vector<float> recall = StatisticsCalculation::calc_recall(cumulative_confusion_matrix);
            std::vector<float> f1 = StatisticsCalculation::calc_f1(cumulative_confusion_matrix);
            Utils::Save::save_metrics(stats_folder + "metrics.txt",accuracy, precision, recall, f1);
        }

        //draw labels on image and save output image
        cv::Mat output_img = img.clone();
        Utils::Visualization::printLabelsOnImage(output_img, true_labels, cv::Scalar(0,255,0), cv::Scalar(0,255,0)); //true labels in green
        Utils::Visualization::printLabelsOnImage(output_img, predicted_labels, cv::Scalar(255,0,0), cv::Scalar(255,0,0)); //predicted labels in red
        auto time_draw_labels_end = std::chrono::steady_clock::now();
        double draw_ms = std::chrono::duration<double, std::milli>(time_draw_labels_end - time_save_pred_ann_end).count();

        Utils::Save::saveImageToFile(images_folder + img_info->get_name() + ".png", output_img);
        auto time_save_img_end = std::chrono::steady_clock::now();
        double save_image_ms = std::chrono::duration<double, std::milli>(time_save_img_end - time_draw_labels_end).count();


        if(visualize){
            cv::Mat vis_img = img.clone();
            //draw true labels in green
            for (const auto& label : true_labels) {
                const std::vector<cv::Rect>& boxes = label.get_bounding_boxes();

                for (const cv::Rect& box : boxes) {
                    cv::rectangle(vis_img, box, cv::Scalar(0, 255, 0), 2);
                    if (label.get_object()) {
                        cv::putText(vis_img, label.get_object()->to_string(), cv::Point(box.x, box.y - 10),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                    }
                }

                
            }
            //draw predicted labels in red
            for (const auto& label : predicted_labels) {

                const std::vector<cv::Rect>& boxes = label.get_bounding_boxes();
                for (const cv::Rect& box : boxes) {
                    cv::rectangle(vis_img, box, cv::Scalar(0, 0, 255), 2);
                    if (label.get_object()) {
                        cv::putText(vis_img, label.get_object()->to_string(), cv::Point(box.x, box.y - 10),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
                     }
                }

                
            }
            //resize image
            float ratio = 0.5;
            cv::resize(vis_img, vis_img, cv::Size(), ratio, ratio, cv::INTER_LINEAR);
            cv::imshow("Detections", vis_img);
            cv::waitKey(0); //display each image for 500 ms
            cv::destroyAllWindows();
        }

        auto time_end = std::chrono::steady_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(time_end - time_start).count();

        double accounted_ms = load_ms + detect_ms + gt_ms + save_pred_ann_ms + draw_ms + save_image_ms;
        double overhead_ms = total_ms - accounted_ms;

        total_load_time  += (time_load_end - time_start);
        total_detect_time += (time_detect_end - time_load_end);
        total_gt_time += (time_gt_end - time_detect_end);
        total_save_annotations_time += (time_save_pred_ann_end - time_gt_end);
        total_draw_time += (time_draw_labels_end - time_save_pred_ann_end);
        total_save_image_time += (time_save_img_end - time_draw_labels_end);
        total_total_time += (time_end - time_start);

        std::cout << " Time for card: " << img_info->get_name()
                << " | load: " << load_ms
                << " | detect: " << detect_ms
                << " | gt: " << gt_ms
                << " | save_ann: " << save_pred_ann_ms
                << " | draw: " << draw_ms
                << " | save_img: " << save_image_ms
                << " | overhead: " << overhead_ms
                << " | total: " << total_ms
                << " ms"
                << std::endl;
        
        if (total_images > 0) {
            Utils::Visualization::printProgressBar(static_cast<float>(idx) / static_cast<float>(total_images), 50, "Processing images: ", "Complete");
        }
    }

    // final cumulative confusion matrix + metrics
    Utils::Save::save_confusion_matrix(stats_folder + "confusion_matrix.txt", cumulative_confusion_matrix);
    float final_accuracy = StatisticsCalculation::calc_accuracy(cumulative_confusion_matrix);
    std::vector<float> precision = StatisticsCalculation::calc_precision(cumulative_confusion_matrix);
    std::vector<float> recall = StatisticsCalculation::calc_recall(cumulative_confusion_matrix);
    std::vector<float> f1 = StatisticsCalculation::calc_f1(cumulative_confusion_matrix);
    Utils::Save::save_metrics(stats_folder + "metrics.txt", final_accuracy, precision, recall, f1);

    // Summary time
    std::cout << "Dataset image path: " << dataset->get_root() << std::endl;
    std::cout << "Dataset annotation path: " << dataset->get_annotation_root() << std::endl;
    std::cout << "Total images processed: " << total_images << std::endl;

    if (total_images > 0) {
        std::cout << "Average load time per image: " << total_load_time.count() / total_images << " ms" << std::endl;
        std::cout << "Average detect time per image: " << total_detect_time.count() / total_images << " ms" << std::endl;
        std::cout << "Average gt time per image: " << total_gt_time.count() / total_images << " ms" << std::endl;
        std::cout << "Average save annotations time per image: " << total_save_annotations_time.count() / total_images << " ms" << std::endl;
        std::cout << "Average draw time per image: " << total_draw_time.count() / total_images << " ms" << std::endl;
        std::cout << "Average save image time per image: " << total_save_image_time.count() / total_images << " ms" << std::endl;
        std::cout << "Average total time per image: " << total_total_time.count() / total_images << " ms" << std::endl;
    }

}
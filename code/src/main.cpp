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
#include "../include/detection/card_detector/YoloCardDetector.h"
#include "../include/Dataset/ImageDataset.h"
#include "../include/Dataset/VideoDataset.h"
#include "../include/Dataset/TemplateDataset.h"
#include "../include/SampleInfo/TemplateInfo.h"
#include "../include/StatisticsCalculation.h"
#include "../include/detection/card_detector/objectClassifiers/featurePipeline/features/FeatureContainer.h"
#include "../include/detection/card_detector/objectClassifiers/featurePipeline/FeaturePipeline.h"
#include "../include/detection/card_detector/objectSegmenters/SimpleContoursCardSegmenter.h"
#include "../include/detection/card_detector/objectSegmenters/WatershedCardSegmenter.h"
#include "../include/VideoWriter.h"

#include <exception>

enum class DETECTION_MODE{
    TEMPLATES = 0,
    MODEL = 1
};


std::unique_ptr<ProcessingMode> create_mode_for_dataset(const std::unique_ptr<Dataset>& dataset, const bool visualize, const DETECTION_MODE detection_mode, TemplateDataset* const template_dataset = nullptr, std::string* const model_path=nullptr);
void iterate_dataset(std::unique_ptr<Dataset>& dataset, const ImageFilter& image_filter, std::unique_ptr<ProcessingMode>& mode, const std::string& output_folder_path, const bool visualize, const int num_classes = 53, const float iou_threshold = 0.5f);
cv::Mat draw_hi_low_count(const cv::Mat& image, const SequentialFrameProcessing& mode, int frame_number);

int main(int argc, char** argv) {

    try{
        if (argc < 5) {
            std::cerr << "Usage: ./program <datasets_path> <template_dataset_path> <models_path> <output_path> <dataset_name_to_use> <method> <visualize_flag>" << std::endl;
            std::cerr << "datasets_path: path to the folder containing the datasets (single_cards and videos folders)" << std::endl;
            std::cerr << "template_dataset_path: path to the folder containing the template cards dataset" << std::endl;
            std::cerr << "models_path: path to the onnx models folder" << std::endl;
            std::cerr << "output_path: path to the folder where the output will be saved" << std::endl;
            std::cerr << "dataset_name_to_use: name of the dataset to use. Possible values are 'single_cards', 'video_images', or 'video'" << std::endl;
            std::cerr << "method: method to use for card detection. Possible values are 'templates' or 'model'" << std::endl;
            std::cerr << "visualize_flag (JUST FOR DEVELOPMENT PURPOSE): whether to visualize the detected images (true/false), optional, default is false" << std::endl;
            return 1;
        }
        std::string datasets_path = argv[1];
        std::string template_dataset_path = argv[2];
        std::string models_path = argv[3];
        std::string output_path = argv[4];
        std::string dataset_to_use = argv[5];
        std::string method = argv[6];

        if (!std::filesystem::exists(datasets_path)) {
            std::cerr << "The datasets path does not exist!" << std::endl;
            return 1;
        }

        if (!std::filesystem::exists(template_dataset_path)) {
            std::cerr << "The template dataset path does not exist!" << std::endl;
            return 1;
        }

        if (!std::filesystem::exists(models_path)) {
            std::cerr << "The models path does not exist!" << std::endl;
            return 1;
        }

        const std::string YOLO_SINGLE_CARD_MODEL_NAME = "yolov11s_single_cards_1280.onnx";
        const std::string YOLO_MULTIPLE_CARDS_MODEL_NAME = "yolov11s_synthetic_1280_20.onnx";

        const std::string SINGLE_CARD_DATASET_NAME = "single_cards";
        const std::string VIDEO_IMAGE_DATASET_NAME = "video_images";
        const std::string VIDEO_DATASET_NAME = "video";


        DETECTION_MODE detection_method;
        if(method == "templates"){
            detection_method = DETECTION_MODE::TEMPLATES;
        } else if(method == "model"){
            detection_method = DETECTION_MODE::MODEL;
        }
        else{
            std::cerr << "The selected method (" << method << ") is not a valid option! Possible options are 'templates' or 'model'" << std::endl;
            return 1;
        }


        std::string dataset_path = "";

        if(dataset_to_use == SINGLE_CARD_DATASET_NAME){
            dataset_path = datasets_path + "/" + "single_cards";

        } else if(dataset_to_use == VIDEO_IMAGE_DATASET_NAME){
            dataset_path = datasets_path + "/" + "video_images";

        }
        else if(dataset_to_use == VIDEO_DATASET_NAME){
            dataset_path = datasets_path + "/" + "videos" + "/" + "video_blue_bg/VideoBlackjack.mp4";
        }
        else{
            std::cerr << "The selected dataset (" << dataset_to_use << ") is not a valid option! Possible options are 'single_cards', 'video_images', 'video'" << std::endl;
            return 1;
        }

        std::string dataset_output_path = output_path + "/" + dataset_to_use;

        if(dataset_to_use == SINGLE_CARD_DATASET_NAME){
            if(detection_method == DETECTION_MODE::TEMPLATES){
                dataset_output_path += "_templates";
            } else if(detection_method == DETECTION_MODE::MODEL){
                dataset_output_path += "_model";
            }
        }
        
        if (std::filesystem::exists(dataset_output_path)) {
            std::cout << "The output path (" << dataset_output_path << ") already exists! Do you want to proceed? (y/n): ";
            char response;
            std::cin >> response;
            if (response != 'y' && response != 'Y') {
                std::cout << "Exiting the program." << std::endl;
                return 0;
            }
            else{
                std::cout << "Overwriting the output path!" << std::endl;
                std::filesystem::remove_all(dataset_output_path);
                std::filesystem::create_directories(dataset_output_path);
            }
        } else {
            std::filesystem::create_directories(dataset_output_path);
            std::cout << "The output path (" << dataset_output_path << ") has been created!" << std::endl;
        }


        

        bool visualize = (argc > 7) ? (std::string(argv[7]) == "true") : false;

        std::cout << "Program started with the following parameters:" << std::endl;
        std::cout << "datasets_path: " << datasets_path << std::endl;
        std::cout << "template_dataset_path: " << template_dataset_path << std::endl;
        std::cout << "models_path: " << models_path << std::endl;
        std::cout << "output_path: " << output_path << std::endl;
        std::cout << "dataset_to_use: " << dataset_to_use << std::endl;
        std::cout << "method: " << method << std::endl;
        std::cout << "visualize: " << (visualize ? "true" : "false") << std::endl;

        constexpr int num_classes = 53; //52 cards + background/no card class
        constexpr float iou_threshold = 0.5f;

        ImageFilter img_filter;

        if(dataset_to_use == SINGLE_CARD_DATASET_NAME){
            //Dataset object creation
            std::unique_ptr<Dataset> single_cards_dataset(new ImageDataset(dataset_path));

            if (detection_method == DETECTION_MODE::TEMPLATES) {
                //TemplateDataset creation 
                TemplateDataset template_dataset(template_dataset_path);
                std::cout << "Template Dataset root: " << template_dataset.get_root() << std::endl;
                std::cout << "Template Dataset loaded with " << template_dataset.size() << " entries." << std::endl;
                if (visualize) {
                    for (auto it = template_dataset.begin(); it != template_dataset.end(); ++it) {
                        const TemplateInfo& sample = dynamic_cast<const TemplateInfo&>(*it);
                        cv::Mat img = template_dataset.load(it);
                        std::cout << "Visualizing template sample: " << sample << std::endl;
                        Utils::Visualization::showImage(img, "Template Card: " + sample.get_name(), 200, 1.0);
                    }
                }
                //depending on the dataset type, create the appropriate card detector
                std::unique_ptr<ProcessingMode> mode = create_mode_for_dataset(single_cards_dataset, visualize, detection_method, &template_dataset);
                //image preprocesing (resize to faster computations)
                img_filter.add_filter("Resize", Filters::resize, 0.25, 0.25); 
                //iterate through dataset and detect each image
                //output is saved into output_path/single_cards
                iterate_dataset(single_cards_dataset, img_filter, mode, dataset_output_path, visualize, num_classes);
            }
            else if (detection_method == DETECTION_MODE::MODEL) {
                std::string model_path = models_path + "/" + YOLO_SINGLE_CARD_MODEL_NAME;
                if (!std::filesystem::exists(model_path)) {
                    std::cerr << "The specified model file does not exist: " << model_path << std::endl;
                    return 1;
                }
                //depending on the dataset type, create the appropriate card detector
                std::unique_ptr<ProcessingMode> mode = create_mode_for_dataset(single_cards_dataset, visualize, detection_method, nullptr, &model_path);
                //image preprocesing (resize to faster computations)
                //img_filter.add_filter("Resize", Filters::resize_to, 1280, 960);
                //iterate through dataset and detect each image
                //output is saved into output_path/single_cards
                iterate_dataset(single_cards_dataset, img_filter, mode, dataset_output_path, visualize, num_classes);
            }
        } else if(dataset_to_use == VIDEO_IMAGE_DATASET_NAME){
            
            //Dataset object creation
            std::unique_ptr<Dataset> multiple_cards_dataset(new ImageDataset(dataset_path));
            
            if(detection_method == DETECTION_MODE::MODEL){
                std::string model_path = models_path + "/" + YOLO_MULTIPLE_CARDS_MODEL_NAME;
                if (!std::filesystem::exists(model_path)) {
                    std::cerr << "The specified model file does not exist: " << model_path << std::endl;
                    return 1;
                }
                //depending on the dataset type, create the appropriate card detector
                std::unique_ptr<ProcessingMode> mode = create_mode_for_dataset(multiple_cards_dataset, visualize, detection_method, nullptr, &model_path);
                //iterate through dataset and detect each image
                //output is saved into output_path/multiple_cards
                //img_filter.add_filter("Resize", Filters::resize_to, 1280, 1280);
                iterate_dataset(multiple_cards_dataset, img_filter, mode, output_path + "/" + dataset_to_use, visualize, num_classes);
            }
            else{
                std::cerr << "The selected method (" << method << ") is not a valid option for the selected dataset (" << dataset_to_use << ")! For the 'video_images' dataset, only the 'model' method is supported." << std::endl;
                return 1;
            }
        } else if(dataset_to_use == VIDEO_DATASET_NAME){
            //Dataset object creation
            constexpr double VIDEO_SAMPLE_FPS = 5.0;
            std::unique_ptr<Dataset> video_dataset(new VideoDataset(dataset_path, false, VIDEO_SAMPLE_FPS));

            if(detection_method == DETECTION_MODE::MODEL){
                std::string model_path = models_path + "/" + YOLO_MULTIPLE_CARDS_MODEL_NAME;
                if (!std::filesystem::exists(model_path)) {
                    std::cerr << "The specified model file does not exist: " << model_path << std::endl;
                    return 1;
                }
                //depending on the dataset type, create the appropriate card detector
                std::unique_ptr<ProcessingMode> mode = create_mode_for_dataset(video_dataset, visualize, detection_method, nullptr, &model_path);
                //iterate through dataset and detect each image
                //output is saved into output_path/video
                iterate_dataset(video_dataset, img_filter, mode, output_path + "/" + VIDEO_DATASET_NAME, visualize, num_classes);
            }
            else{
                std::cerr << "The selected method (" << method << ") is not a valid option for the selected dataset (" << dataset_to_use << ")! For the 'video' dataset, only the 'model' method is supported." << std::endl;
                return 1;
            }
        }
    }catch (const std::runtime_error& e) {
        std::cerr << "RUNTIME ERROR: " << e.what() << std::endl;
        return -1;
    }
    catch (const cv::Exception& e) {
        std::cerr << "OPENCV ERROR: " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception& e) {
        std::cerr << "GENERIC ERROR: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}

std::unique_ptr<ProcessingMode> create_mode_for_dataset(const std::unique_ptr<Dataset>& dataset, const bool visualize, const DETECTION_MODE detection_mode, TemplateDataset* const template_dataset, std::string* const model_path) {
    std::unique_ptr<CardDetector> card_detector = nullptr;

    switch(detection_mode){
    case DETECTION_MODE::TEMPLATES:
        if (!template_dataset){
            throw std::runtime_error("Template dataset is required but pointer to it is nullptr!");
        }
        card_detector = std::make_unique<SegmentationClassificationCardDetector>(std::make_unique<MaskCardDetector>(PipelinePreset::DEFAULT, MaskType::CONVEX_HULL, visualize), std::make_unique<FeaturePipeline>(ExtractorType::FeatureDescriptorAlgorithm::SIFT, 1, *template_dataset), std::make_unique<WatershedCardSegmenter>(), visualize);
        break;
    case DETECTION_MODE::MODEL:
        if (!model_path){
            throw std::runtime_error("Model is required but pointer to its path is nullptr!");
        }
        card_detector = std::make_unique<YoloCardDetector>(*model_path, visualize);
        break;
    default:
        throw std::runtime_error("Detection mode not specified!");
    }

    if (dataset->is_sequential()) {
        return std::make_unique<SequentialFrameProcessing>(std::move(card_detector), visualize, 5.0);
    } else {
        return std::make_unique<SingleFrameProcessing>(std::move(card_detector), visualize);
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
    int object_count = 0;

    cv::Mat cumulative_confusion_matrix = cv::Mat::zeros(num_classes, num_classes, CV_32S);
    const int save_cm_every = 10;

    std::unique_ptr<VideoWriter> video_writer;
    if(dataset->is_sequential()){
        // Output at 15 FPS for slightly faster than realtime playback
        video_writer = std::make_unique<VideoWriter>(output_folder_path+"/"+dataset->get_root().filename().string(), 5);
    }

    int frame_number = 0;
    std::set<int> encountered_classes;
    float all_objects_mean_iou = 0.0f;
    for (auto it = dataset->begin(); it != dataset->end(); ++it) {

        //vectors to hold predicted and true labels for the current image
        std::vector<Label> predicted_labels;
        std::vector<Label> true_labels;

        double load_ms = 0, detect_ms = 0, gt_ms = 0, save_pred_ann_ms = 0, draw_ms = 0, save_image_ms = 0;

        auto image_loop_start = std::chrono::steady_clock::now();
        auto step_start = image_loop_start;

        step_start = std::chrono::steady_clock::now();
        //load and filter image
        SampleInfo* img_info = &(*it);
        cv::Mat img = dataset->load(it);

        if(img.empty()){
            // Skip unreadable frames (common at video end due to corruption/encoding issues)
            continue;
        }

        img = image_filter.apply_filters(img);
        load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - step_start).count();

        step_start = std::chrono::steady_clock::now();
        //detects cards in image and adds the result of the detection to the vector
        predicted_labels = mode->detect_image(img);
        detect_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - step_start).count();

        step_start = std::chrono::steady_clock::now();
        //saves the predicted labels to a file
        Utils::Save::saveLabelsToYoloFile(annotations_folder + img_info->get_name() + ".txt", predicted_labels, img.cols, img.rows);
        save_pred_ann_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - step_start).count();

        if(dataset->get_has_annotations()){
            step_start = std::chrono::steady_clock::now();
            //load ground truth labels
            true_labels = Loader::Annotation::load_yolo_image_annotations(img_info->get_pathLabel(), img.cols, img.rows);
            for (const auto& label : true_labels) {
                if (const CardType* card = dynamic_cast<const CardType*>(label.get_object())) {
                    encountered_classes.insert(Yolo_index_codec::card_to_yolo_index(*card));
                }
            }

            //update cumulative confusion matrix
            cumulative_confusion_matrix += StatisticsCalculation::calc_confusion_matrix(true_labels, predicted_labels, num_classes, iou_threshold);
            float accuracy = StatisticsCalculation::calc_accuracy(cumulative_confusion_matrix);
            if (save_cm_every > 0 && (idx % save_cm_every == 0)) {
                
                Utils::Save::save_confusion_matrix(stats_folder + "confusion_matrix.txt", cumulative_confusion_matrix);
                
                std::vector<float> cards_iou = StatisticsCalculation::calc_image_meanIoU(true_labels, predicted_labels);
                
                for (size_t n = 0; n < cards_iou.size(); ++n) {
                    float a_n = cards_iou[n];
                    all_objects_mean_iou += (a_n - all_objects_mean_iou) / static_cast<float>(object_count + n + 1);
                }
                object_count += cards_iou.size();
                
                std::vector<float> precision = StatisticsCalculation::calc_precision(cumulative_confusion_matrix);
                std::vector<float> recall = StatisticsCalculation::calc_recall(cumulative_confusion_matrix);
                std::vector<float> f1 = StatisticsCalculation::calc_f1(cumulative_confusion_matrix);
                Utils::Save::save_metrics(stats_folder + "metrics.txt", accuracy, all_objects_mean_iou, precision, recall, f1, encountered_classes);
            }

            gt_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - step_start).count();
        }

        step_start = std::chrono::steady_clock::now();
        //draw labels on image and save output image
        cv::Mat output_img = img.clone();
        if(dataset->get_has_annotations()){
            Utils::Visualization::printLabelsOnImage(output_img, true_labels, cv::Scalar(0,0,0), cv::Scalar(0,0,0)); //true labels in black
        }
        Utils::Visualization::printLabelsOnImageHiLo(output_img, predicted_labels); //Hi-Lo color-coded bounding boxes

        if(dataset->is_sequential()){
            auto* seq_mode = dynamic_cast<SequentialFrameProcessing*>(mode.get());
            output_img = draw_hi_low_count(output_img, *seq_mode, frame_number);
            video_writer->addFrame(output_img);
        }

        draw_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - step_start).count();

        step_start = std::chrono::steady_clock::now();
        Utils::Save::saveImageToFile(images_folder + img_info->get_name() + ".png", output_img);
        save_image_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - step_start).count();

        if(visualize){
            Utils::Visualization::showImage(output_img, "Detections", 500, 1);
        }

        auto image_loop_end = std::chrono::steady_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(image_loop_end - image_loop_start).count();

        total_load_time             += std::chrono::duration<double, std::milli>(load_ms);
        total_detect_time           += std::chrono::duration<double, std::milli>(detect_ms);
        total_gt_time               += std::chrono::duration<double, std::milli>(gt_ms);
        total_save_annotations_time += std::chrono::duration<double, std::milli>(save_pred_ann_ms);
        total_draw_time             += std::chrono::duration<double, std::milli>(draw_ms);
        total_save_image_time       += std::chrono::duration<double, std::milli>(save_image_ms);
        total_total_time            += std::chrono::duration<double, std::milli>(total_ms);

        double accounted_ms = load_ms + detect_ms + gt_ms + save_pred_ann_ms + draw_ms + save_image_ms;
        double overhead_ms = total_ms - accounted_ms;

        std::stringstream ss;
        ss << std::fixed << std::setprecision(4);
        ss << "\n Time for card: " << img_info->get_name()
            << "\n | load:      " << load_ms << " ms"
            << "\n | detect:    " << detect_ms << " ms"
            << "\n | gt:        " << (dataset->get_has_annotations() ? std::to_string(gt_ms) : "N/A") << " ms"
            << "\n | save_ann:  " << save_pred_ann_ms << " ms"
            << "\n | draw:      " << draw_ms << " ms"
            << "\n | save_img:  " << save_image_ms << " ms"
            << "\n | overhead:  " << overhead_ms << " ms"
            << "\n | total:     " << total_ms << " ms\n";

        std::string time_output = ss.str();

        idx++;
        
        if (total_images > 0) {
            Utils::Visualization::printProgressBar(static_cast<float>(idx) / static_cast<float>(total_images), 50, "Processing: ", "Complete"+time_output);
        }
        
        frame_number++;
    }

    if(dataset->is_sequential()){
        video_writer->close();
        std::cout << "\nVideo saved to: " << video_writer->get_output_path() << std::endl;
    }

    // final cumulative confusion matrix + metrics
    Utils::Save::save_confusion_matrix(stats_folder + "confusion_matrix.txt", cumulative_confusion_matrix);
    float final_accuracy = StatisticsCalculation::calc_accuracy(cumulative_confusion_matrix);
    std::vector<float> precision = StatisticsCalculation::calc_precision(cumulative_confusion_matrix);
    std::vector<float> recall = StatisticsCalculation::calc_recall(cumulative_confusion_matrix);
    std::vector<float> f1 = StatisticsCalculation::calc_f1(cumulative_confusion_matrix);
    Utils::Save::save_metrics(stats_folder + "metrics.txt", final_accuracy, all_objects_mean_iou, precision, recall, f1, encountered_classes);

    std::cout << "Total images processed: " << total_images << std::endl;

    if (total_images > 0) {
        std::cout << "Average load time per image: " << total_load_time.count() / total_images << " ms" << std::endl;
        std::cout << "Average detect time per image: " << total_detect_time.count() / total_images << " ms" << std::endl;
        std::cout << "Average gt time per image: " << (dataset->get_has_annotations() ? std::to_string(total_gt_time.count() / total_images) : "N/A")  << " ms" << std::endl;
        std::cout << "Average save annotations time per image: " << total_save_annotations_time.count() / total_images << " ms" << std::endl;
        std::cout << "Average draw time per image: " << total_draw_time.count() / total_images << " ms" << std::endl;
        std::cout << "Average save image time per image: " << total_save_image_time.count() / total_images << " ms" << std::endl;
        std::cout << "Average total time per image: " << total_total_time.count() / total_images << " ms" << std::endl;
    }

}

cv::Mat draw_hi_low_count(const cv::Mat& image, const SequentialFrameProcessing& mode, int frame_number){
    cv::Mat output_img = image.clone();
    // Display Hi-Lo count
    int count = mode.get_running_count();
    std::string count_text = "Hi-Lo Count: " + std::to_string(count);
    cv::putText(output_img, count_text, cv::Point(20, 50), 
        cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255, 255, 255), 3);
    cv::putText(output_img, count_text, cv::Point(20, 50), 
        cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 2);

    // Display frame number
    cv::putText(output_img, "Frame: " + std::to_string(frame_number), cv::Point(20, 100), 
        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);

    // Display tracked cards status (card ID, state, and physical card count)
    const std::map<std::string, CardTracker::TrackedCard>& tracked_cards = mode.get_tracked_cards();
    int y_offset = 140;
    for (std::map<std::string, CardTracker::TrackedCard>::const_iterator it = tracked_cards.begin(); it != tracked_cards.end(); ++it) {
        const std::string& card_id = it->first;
        const CardTracker::TrackedCard& tracked = it->second;
        std::string state_str;
        cv::Scalar color;
        switch (tracked.state) {
            case CardTracker::CardState::CANDIDATE: 
                state_str = "CAND"; color = cv::Scalar(0, 255, 255); break;
            case CardTracker::CardState::CONFIRMED: 
                state_str = "CONF"; color = cv::Scalar(0, 255, 0); break;
            case CardTracker::CardState::OCCLUDED: 
                state_str = "OCCL"; color = cv::Scalar(0, 165, 255); break;
            default: continue;
        }
        // Show card ID, state, and count (e.g., "AS [CONF] x1" or "AS [CONF] x2")
        std::string info = card_id + " [" + state_str + "] x" + std::to_string(tracked.confirmed_card_count);
        cv::putText(output_img, info, cv::Point(20, y_offset), 
            cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        y_offset += 25;
    }
    return output_img;
}
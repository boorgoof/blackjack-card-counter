#include <iostream>
#include <cstddef>
#include <opencv2/opencv.hpp>
#include "../include/Dataset/Dataset.h"
#include "../include/VideoWriter.h"
#include "../include/Dataset/ImageDataset.h"
#include "../include/Dataset/VideoDataset.h"
#include "../include/SampleInfo/SampleInfo.h"
#include "../include/detection/card_detector/YoloCardDetector.h"
#include "../include/detection/SingleFrameProcessing.h"
#include "../include/Utils.h"
#include <memory>
#include <vector>

int main() {
    
    // Create dataset instance (polymorphic)
    const bool use_video_dataset = true;
    std::unique_ptr<Dataset> dataset;
    if (use_video_dataset) {
        dataset = std::make_unique<VideoDataset>(std::string("../data/datasets/VideoBlackjack.mp4"));

    } else {
        dataset = std::make_unique<ImageDataset>(std::string("../data/datasets/single_cards/Images/Images"), std::string("../data/datasets/single_cards/YOLO_Annotations/YOLO_Annotations/"));
    }

    std::unique_ptr<YoloCardDetector> card_detector = std::make_unique<YoloCardDetector>("../DL_approach/yolov11s_synthetic_1280.onnx");
    std::unique_ptr<SingleFrameProcessing> mode = std::make_unique<SingleFrameProcessing>(std::move(card_detector));

    VideoWriter videoW = VideoWriter("output_video.avi", 30.0);

    for (auto it = dataset->begin(); it != dataset->end(); ++it) {


        SampleInfo* img_info = &(*it);
        cv::Mat img = dataset->load(it);

        std::vector<Label> predicted_labels = mode->detect_image(img); 
        cv::Mat output_img = img.clone();
        Utils::Visualization::printLabelsOnImage(output_img, predicted_labels, cv::Scalar(0,255,0), cv::Scalar(0,255,0));

        videoW.addFrame(output_img);
        //cv::imshow("Detections", output_img);
        //cv::waitKey(0); // Wait for a key press to proceed to the next
        

    }
    
    cv::destroyAllWindows();
    return 0;
}

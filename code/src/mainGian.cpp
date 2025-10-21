#include <iostream>
#include <cstddef>
#include <opencv2/opencv.hpp>
#include "../include/card_detector/RoughCardDetector.h"
#include "../include/Dataset/Dataset.h"
#include "../include/Dataset/ImageDataset.h"
#include "../include/Dataset/VideoDataset.h"
#include "../include/SampleInfo/SampleInfo.h"
#include <memory>
#include <vector>

int main() {
    // Initialize the card detector with default preset and polygon mask type
    vision::RoughCardDetector detector{vision::PipelinePreset::DEFAULT, vision::MaskType::POLYGON};
    
    // Create dataset instance (polymorphic)
    const bool use_video_dataset = true;
    std::unique_ptr<Dataset> dataset;
    if (use_video_dataset) {
        dataset = std::make_unique<VideoDataset>(std::string("../data/VideoBlackjack.mp4"));
    } else {
        dataset = std::make_unique<ImageDataset>(std::string("../data/datasets/single_cards/Images/Images"), std::string("../data/datasets/single_cards/YOLO_Annotations/YOLO_Annotations/"));
    }

    for (auto it = dataset->begin(); it != dataset->end(); ++it) {
        SampleInfo& sample = *it;
        cv::Mat imageFiles = dataset->load(it);
        
        if (imageFiles.empty()) {
            std::cout << "Could not load sample: " << sample.get_pathSample() << std::endl;
            continue;
        }

        cv::Mat currentMask = detector.getMask(imageFiles);
            
        // Method 2: Change mask type and get different masks
        detector.loadMaskPreset(vision::MaskType::POLYGON);
        cv::Mat polygonMask = detector.getMask(imageFiles);
            
        detector.loadMaskPreset(vision::MaskType::CONVEX_HULL);
        cv::Mat convexHullMask = detector.getMask(imageFiles);
            
        detector.loadMaskPreset(vision::MaskType::BOUNDING_BOX);
        cv::Mat boundingBoxMask = detector.getMask(imageFiles);

        cv::Mat displayWithBBox = imageFiles.clone();
        std::vector<cv::Point> nonZeroPoints;
        cv::findNonZero(convexHullMask, nonZeroPoints);
        if (!nonZeroPoints.empty()) {
            cv::Rect bbox = cv::boundingRect(nonZeroPoints);
            cv::rectangle(displayWithBBox, bbox, cv::Scalar(0, 255, 0), 2);
        }

        // Display results
        cv::imshow("Original Frame", imageFiles);
        cv::imshow("Frame + Convex Hull BBox", displayWithBBox);
        cv::imshow("Polygon mask", polygonMask);
        cv::imshow("Convex Hull Mask", convexHullMask);
        cv::imshow("Bounding Box Mask", boundingBoxMask);

        // For debugging: print mask sizes
        std::cout << "Polygon mask size: " << polygonMask.size() << std::endl;
        std::cout << "Convex hull mask size: " << convexHullMask.size() << std::endl;
        std::cout << "Bounding box mask size: " << boundingBoxMask.size() << std::endl;
        
        cv::waitKey(0);
    }
    
    cv::destroyAllWindows();
    return 0;
}

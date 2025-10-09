#include <iostream>
#include <opencv2/opencv.hpp>
#include "../include/card_detector/RoughCardDetector.h"
#include "../include/Dataset.h"
#include "../include/ImageInfo.h"

int main() {
    // Initialize the card detector with default preset and polygon mask type
    vision::RoughCardDetector detector{vision::PipelinePreset::DEFAULT, vision::MaskType::POLYGON};
    
    // Create dataset instance
    int i = 0;
    Dataset dataset = (i == 0) ? 
        Dataset(std::string("../data/datasets/videos/images/"), std::string("../data/datasets/videos/labels/"), false) :
        Dataset(std::string("../data/datasets/single_cards/Images/Images"), std::string("../data/datasets/single_cards/YOLO_Annotations/YOLO_Annotations/"), false);

    for (auto it = dataset.begin(); it != dataset.end(); ++it) {
        ImageInfo& image = *it;
        cv::Mat imageFiles = cv::imread(image.get_pathImage());
        
        if (imageFiles.empty()) {
            std::cout << "Could not load image: " << image.get_pathImage() << std::endl;
            continue;
        }

        // === ADD MASK TESTING ===
        
        // Method 1: Use the detector's current mask type (set in constructor)
        cv::Mat currentMask = detector.getMask(imageFiles);
        
        // Method 2: Change mask type and get different masks
        detector.loadMaskPreset(vision::MaskType::POLYGON);
        cv::Mat polygonMask = detector.getMask(imageFiles);
        
        detector.loadMaskPreset(vision::MaskType::CONVEX_HULL);
        cv::Mat convexHullMask = detector.getMask(imageFiles);
        
        detector.loadMaskPreset(vision::MaskType::BOUNDING_BOX);
        cv::Mat boundingBoxMask = detector.getMask(imageFiles);
        
        // Display results
        cv::imshow("Original Image", imageFiles);
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

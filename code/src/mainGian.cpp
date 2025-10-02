#include <iostream>
#include <opencv2/opencv.hpp>
#include "../include/card_detector/RoughCardDetector.h"
#include "../include/Dataset.h"
#include "../include/ImageInfo.h"

int main() {
    // Initialize the card detector
    vision::RoughCardDetector detector{vision::PipelinePreset::DEFAULT};
    
    // Create dataset instance
    int i = 1;
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
        
        // Test getCardsPolygon
        std::vector<std::vector<cv::Point>> polygons = detector.getCardsPolygon(imageFiles);
        cv::Mat polygonImage = imageFiles.clone();
        
        // Draw polygons
        for (size_t i = 0; i < polygons.size(); ++i) {
            cv::Scalar color(0, 255, 0); // Green color
            cv::polylines(polygonImage, polygons[i], true, color, 2);
        }

        // Test getCardsConvexHulls
        std::vector<std::vector<cv::Point>> convexHulls = detector.getCardsConvexHulls(imageFiles);
        cv::Mat convexHullImage = imageFiles.clone();
        for (size_t i = 0; i < convexHulls.size(); ++i) {
            cv::Scalar color(0, 0, 255); // Red color for convex hull
            cv::polylines(convexHullImage, convexHulls[i], true, color, 2);
        }
        
        // Test getCardsBoundingBox
        std::vector<cv::Rect> boundingBoxes = detector.getCardsBoundingBox(imageFiles);
        cv::Mat boundingBoxImage = imageFiles.clone();
        
        // Draw bounding boxes
        for (size_t i = 0; i < boundingBoxes.size(); ++i) {
            cv::Scalar color(255, 0, 0); // Blue color for bounding boxes
            cv::rectangle(boundingBoxImage, boundingBoxes[i], color, 2);
        }

        // === ADD MASK TESTING ===
        
        // Test getCardPolygonMask
        cv::Mat polygonMask = detector.getCardPolygonMask(imageFiles);
        
        // Test getCardsConvexHullsMask
        cv::Mat convexHullMask = detector.getCardsConvexHullsMask(imageFiles);
        
        // Test getBoundingBoxesMask
        cv::Mat boundingBoxMask = detector.getBoundingBoxesMask(imageFiles);
        
        // Display results
        cv::imshow("Original Image", imageFiles);
        cv::imshow("Polygon mask", polygonMask);
        cv::imshow("Polygons", polygonImage);
        cv::imshow("Convex Hull Mask", convexHullMask);
        cv::imshow("Convex Hulls", convexHullImage);
        cv::imshow("Bounding Box Mask", boundingBoxMask);
        cv::imshow("Bounding Boxes", boundingBoxImage);

        
        
        // Display masks
        if (!polygonMask.empty()) {
            cv::imshow("Polygon Mask", polygonMask);
        }
        if (!convexHullMask.empty()) {
            cv::imshow("Convex Hull Mask", convexHullMask);
        }
        if (!boundingBoxMask.empty()) {
            cv::imshow("Bounding Box Mask", boundingBoxMask);
        }
        
        std::cout << "Found " << polygons.size() << " card polygons" << std::endl;
        std::cout << "Polygon mask size: " << polygonMask.size() << std::endl;
        std::cout << "Convex hull mask size: " << convexHullMask.size() << std::endl;
        std::cout << "Bounding box mask size: " << boundingBoxMask.size() << std::endl;
        
        cv::waitKey(0);
    }
    
    cv::destroyAllWindows();
    return 0;
}

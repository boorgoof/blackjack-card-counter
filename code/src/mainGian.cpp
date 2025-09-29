#include <iostream>
#include <opencv2/opencv.hpp>
#include "../include/card_detector/RoughCardDetector.h"
#include "../include/Dataset.h"
#include "../include/ImageInfo.h"

int main() {
    // Initialize the card detector
    RoughCardDetector detector = RoughCardDetector();
    
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
        
        // Test getCardsPolygon
        std::vector<std::vector<cv::Point>> polygons = detector.getCardsPolygon(imageFiles);
        cv::Mat polygonImage = imageFiles.clone();
        
        // Draw polygons
        for (size_t i = 0; i < polygons.size(); ++i) {
            cv::Scalar color(0, 255, 0); // Green color
            
            // Draw actual polygon using all points
            cv::polylines(polygonImage, polygons[i], true, color, 2);
            
        }

        // Test getConvexHulls
        std::vector<std::vector<cv::Point>> convexHulls = detector.getConvexHulls(imageFiles);
        cv::Mat convexHullImage = imageFiles.clone();
        for (size_t i = 0; i < convexHulls.size(); ++i) {
            cv::Scalar color(0, 0, 255); // Red color for convex hull
            cv::polylines(convexHullImage, convexHulls[i], true, color, 2);
        }

        // Display convex hull image
        cv::imshow("Convex Hulls", convexHullImage);
        
        // Test getCardsBoundingBox
        std::vector<cv::Rect> boundingBoxes = detector.getCardsBoundingBox(imageFiles);
        cv::Mat boundingBoxImage = imageFiles.clone();
        
        // Draw bounding boxes
        for (size_t i = 0; i < boundingBoxes.size(); ++i) {
            cv::Scalar color(255, 0, 0); // Blue color for bounding boxes
            cv::rectangle(boundingBoxImage, boundingBoxes[i], color, 2);
        }
        // Display results
        cv::imshow("Original Image", imageFiles);
        cv::imshow("Polygons", polygonImage);
        cv::imshow("Bounding Boxes", boundingBoxImage);
        
        std::cout << "Found " << polygons.size() << " card polygons" << std::endl;
        
        cv::waitKey(0);
    }
    
    cv::destroyAllWindows();
    return 0;
}
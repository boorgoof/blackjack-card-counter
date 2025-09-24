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
    // Access the ImageInfo object
    ImageInfo& image = *it;
    // Use the image object

    cv::Mat imageFiles = cv::imread(image.get_pathImage());
    detector.getCardsPolygon(imageFiles);
}
    cv::destroyAllWindows();
    return 0;
}
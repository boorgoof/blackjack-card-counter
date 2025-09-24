#ifndef ROUGHCARDDETECTOR_H
#define ROUGHCARDDETECTOR_H

#include "../Dataset.h"

class RoughCardDetector{
    public:
        RoughCardDetector();
        std::vector<std::vector<cv::Point>> getCardsPolygon(const cv::Mat& originalImage);
    private:
        void removeNoise(cv::Mat& image) { cv::GaussianBlur(image, image, cv::Size(5, 5), 0); }
        cv::Mat whiteTreshold(const cv::Mat& image);
        void filterBySize(cv::Mat& mask, int minArea);
};


#endif
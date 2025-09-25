#ifndef ROUGHCARDDETECTOR_H
#define ROUGHCARDDETECTOR_H

#include "../Dataset.h"

class RoughCardDetector{
    public:
        RoughCardDetector();
        cv::Mat getCardsMask(const cv::Mat& originalImage);
        std::vector<std::vector<cv::Point>> getCardsPolygon(const cv::Mat& originalImage);
        std::vector<std::vector<cv::Point>> getConvexHulls(const cv::Mat& originalImage);
        cv::Mat getCardsBoundingBox(const cv::Mat& originalImage);
    private:
        cv::Mat whiteTreshold(const cv::Mat& image);
        void filterBySize(cv::Mat& mask, int minArea);
        void morphologicalCleanup(cv::Mat& mask, int openSize = 5, int closeSize = 7);
};


#endif
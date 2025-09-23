#ifndef ROUGHCARDDETECTOR_H
#define ROUGHCARDDETECTOR_H

#include "../Dataset.h"

class RoughCardDetector{
    public:
        RoughCardDetector() = delete;
        RoughCardDetector(cv::Mat img);
    private:
        cv::Mat img_;
};


#endif
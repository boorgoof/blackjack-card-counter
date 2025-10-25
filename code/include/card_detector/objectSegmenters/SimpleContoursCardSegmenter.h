#ifndef SIMPLE_CONTOURS_CARD_SEGMENTER_H
#define SIMPLE_CONTOURS_CARD_SEGMENTER_H

#include "ObjectSegmenter.h"
#include <opencv2/opencv.hpp>


class SimpleContoursCardSegmenter : public ObjectSegmenter {
private:
   
    struct Params {
        double minCardArea = 1000.0;     
        int minContourPoints = 5;          
    };
    Params params_;

public:
    
    SimpleContoursCardSegmenter();

   
    ~SimpleContoursCardSegmenter() override = default;

    std::vector<cv::RotatedRect> segment_objects(const cv::Mat& src_img, const cv::Mat& src_mask) override;

   
    void setMinCardArea(double area) { params_.minCardArea = area; }
    void setMinContourPoints(int points) { params_.minContourPoints = points; }

    double getMinCardArea() const { return params_.minCardArea; }
    int getMinContourPoints() const { return params_.minContourPoints; }
};

#endif // SIMPLE_CONTOURS_CARD_SEGMENTER_H
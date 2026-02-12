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

    /**
     * @brief segment objects in an image given a mask
     * @param src_img the image to segment objects from (in this case, the image is useless since we are only using the mask to find contours)
     * @param src_mask the mask to apply to the image to select the regions to segment
     * @return a vector of contours, where each contour is a vector of points representing the boundary of a segmented object
     */
    std::vector<std::vector<cv::Point>> segment_objects(const cv::Mat& src_img, const cv::Mat& src_mask) override;

    void setMinCardArea(double area) { params_.minCardArea = area; }
    void setMinContourPoints(int points) { params_.minContourPoints = points; }

    double getMinCardArea() const { return params_.minCardArea; }
    int getMinContourPoints() const { return params_.minContourPoints; }
};

#endif // SIMPLE_CONTOURS_CARD_SEGMENTER_H
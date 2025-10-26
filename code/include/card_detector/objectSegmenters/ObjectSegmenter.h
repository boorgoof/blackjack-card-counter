#ifndef OBJECT_SEGMENTER_H
#define OBJECT_SEGMENTER_H

#include <opencv2/opencv.hpp>
#include "../../Label.h"
#include "../../Dataset/Dataset.h"



class ObjectSegmenter{

public:

    ObjectSegmenter() = default;

    ObjectSegmenter(ObjectSegmenter&&) = delete;
    ObjectSegmenter& operator=(ObjectSegmenter&&) = delete; 
    virtual ~ObjectSegmenter() = 0;

    virtual std::vector<cv::RotatedRect> segment_objects(const cv::Mat& src_img, const cv::Mat &src_mask) = 0;
    
    void set_method_name(const std::string& method_name) { this->method_name = method_name; }
    const std::string& get_method_name() const { return this->method_name; }
    
private:
    std::string method_name;
};

#endif // OBJECT_SEGMENTERR_H
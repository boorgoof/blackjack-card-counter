//Federico Meneghetti

#ifndef OBJECT_SEGMENTER_H
#define OBJECT_SEGMENTER_H

#include <opencv2/opencv.hpp>
#include "../../../Label.h"
#include "../../../Dataset/Dataset.h"



class ObjectSegmenter {
public:
   
    ObjectSegmenter() = default;
    
    ObjectSegmenter(ObjectSegmenter&&) = delete;
    ObjectSegmenter& operator=(ObjectSegmenter&&) = delete;
    

    /**
     * @brief segment objects in an image given a mask
     * @param src_img the image to segment objects from
     * @param src_mask the mask to apply to the image to select the regions to segment
     * @return a vector of contours, where each contour is a vector of points representing the boundary of a segmented object
     */
    virtual std::vector<std::vector<cv::Point>> segment_objects(const cv::Mat& src_img, const cv::Mat& src_mask) = 0;
    virtual ~ObjectSegmenter() = 0;

    /**
     * @brief set the name of the segmentation method 
     * @param method_name the name of the segmentation method to set
     */
    void set_method_name(const std::string& method_name) { this->method_name = method_name; }

    /**
     * @brief get the name of the segmentation method 
     */
    const std::string& get_method_name() const { return this->method_name;}

private:
    std::string method_name;
};

#endif // OBJECT_SEGMENTER_H



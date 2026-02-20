//Federico Meneghetti

#ifndef WATERSHED_OBJECT_SEGMENTER_H
#define WATERSHED_OBJECT_SEGMENTER_H

#include "ObjectSegmenter.h"
#include <opencv2/opencv.hpp>

class WatershedCardSegmenter : public ObjectSegmenter {
public:

    WatershedCardSegmenter();
    ~WatershedCardSegmenter() override = default;

    std::vector<std::vector<cv::Point>>

    /**
     * @brief segment objects in an image given a mask using the watershed algorithm with distance transform and markers
     * 
     * @param src_img the image to segment objects from
     * @param src_mask the mask to apply to the image to select the regions to segment
     * @return a vector of contours, where each contour is a vector of points representing the boundary of a segmented object
     */
    segment_objects(const cv::Mat& src_img, const cv::Mat& src_mask) override;

   
private:
    

    /**
     * @brief find peaks in the distance transform image to serve as markers for watershed segmentation
     * @param dist The distance transform image where each pixel value represents the distance to the nearest background pixel 
     * @return A binary image where peaks are marked as white and non-peaks as black
     */
    cv::Mat peaksFromDistance(const cv::Mat& dist);

    /**
     * @brief Generates watershed markers from the identified peaks
     * @param peaks A binary image where peaks are marked as white and non-peaks as black
     * @param marker_contours  the contours of the identified markers
     * @return A marker image where each marker is labeled with a unique integer value
     */
    cv::Mat markersFromPeaks(const cv::Mat& peaks, std::vector<std::vector<cv::Point>>& marker_contours);
    
    /**
     * @brief Applies watershed segmentation to the input image using the markers
     * @param srcColor The original color image to be segmented
     * @param markers The marker image where each marker is labeled with a unique integer value
     * @param marker_contours The contours of the identified markers
     */
    cv::Mat watershedSegmentation(const cv::Mat& srcColor, cv::Mat& markers,const std::vector<std::vector<cv::Point>>& marker_contours);
};

#endif // WATERSHED_OBJECT_SEGMENTER_H
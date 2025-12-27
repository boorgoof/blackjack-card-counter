#ifndef CARDPROJECTION_H
#define CARDPROJECTION_H

#include <opencv2/opencv.hpp>
#include <vector>

namespace CardProjection {
  
    cv::Mat getPerspectiveTranform(const cv::Mat& image, const std::vector<cv::Point>& contour);
    void compute_two_opposite_corners_bboxes(const cv::Mat& H_inv, int cardWidth, int cardHeight,  cv::Rect& bbox1,  cv::Rect& bbox2, float cornerWRatio = 0.20f, float cornerHRatio = 0.25f);
    void convertToIntPoints(const std::vector<cv::Point2f>& foalPoints, std::vector<cv::Point>& intPoints);

}

#endif // CARDPROJECTION_H

#include "../../../include/detection/card_detector/CardProjection.h"
#include <opencv2/imgproc.hpp>

namespace CardProjection {

cv::Mat getPerspectiveTranform(const cv::Mat& image, const std::vector<cv::Point>& contour) {
   
    std::vector<cv::Point> approx;
    cv::approxPolyDP(contour, approx, 0.02 * cv::arcLength(contour, true), true);
    
    std::vector<cv::Point2f> corners;
    if (approx.size() == 4) {
        for (const cv::Point& pt : approx) {
            corners.push_back(cv::Point2f(pt.x, pt.y));
        }
    } else {
        cv::RotatedRect rect = cv::minAreaRect(contour);
        cv::Point2f pts[4];
        rect.points(pts);
        for (int i = 0; i < 4; i++) {
            corners.push_back(pts[i]);
        }
    }
    
    cv::Point2f center(0, 0);
    for (const cv::Point2f& pt : corners) {
        center += pt;
    }
    center /= 4.0f;
    
    std::vector<std::pair<double, cv::Point2f>> anglePoints;
    for (const cv::Point2f& pt : corners) {
        anglePoints.push_back({std::atan2(pt.y - center.y, pt.x - center.x), pt});
    }
    std::sort(anglePoints.begin(), anglePoints.end(),
              [](const std::pair<double, cv::Point2f>& a, const std::pair<double, cv::Point2f>& b) {
                  return a.first < b.first;
              });
    
    std::vector<cv::Point2f> orderedCorners;
    for (const std::pair<double, cv::Point2f>& ap : anglePoints) {
        orderedCorners.push_back(ap.second);
    }
    
    int topLeftIdx = 0;
    float minSum = orderedCorners[0].x + orderedCorners[0].y;
    for (int i = 1; i < 4; i++) {
        float sum = orderedCorners[i].x + orderedCorners[i].y;
        if (sum < minSum) {
            minSum = sum;
            topLeftIdx = i;
        }
    }
    
    std::vector<cv::Point2f> srcPoints;
    for (int i = 0; i < 4; i++) {
        srcPoints.push_back(orderedCorners[(topLeftIdx + i) % 4]);
    }
    
    float avgWidth = (cv::norm(srcPoints[0] - srcPoints[1]) + cv::norm(srcPoints[2] - srcPoints[3])) / 2.0f;
    float avgHeight = (cv::norm(srcPoints[1] - srcPoints[2]) + cv::norm(srcPoints[3] - srcPoints[0])) / 2.0f;
    
    if (avgWidth > avgHeight) {
        std::rotate(srcPoints.begin(), srcPoints.begin() + 1, srcPoints.end());
    }
    
    std::vector<cv::Point2f> dstPoints = {{0, 0}, {249, 0}, {249, 349}, {0, 349}};
    cv::Mat transform = cv::getPerspectiveTransform(srcPoints, dstPoints);
    return transform; 
   
}

void compute_two_opposite_corners_bboxes(const cv::Mat& H_inv,  int cardWidth, int cardHeight, cv::Rect& bbox1, cv::Rect& bbox2, float cornerWidthRatio, float cornerHeightRatio){
    
    // initial empty rects
    bbox1 = cv::Rect();
    bbox2 = cv::Rect();

    if (H_inv.empty() || cardWidth <= 0 || cardHeight <= 0)
        return;

    int cornerWidth  = static_cast<int>(cardWidth * cornerWidthRatio);
    int cornerHeight = static_cast<int>(cardHeight * cornerHeightRatio);

    cornerWidth  = std::max(cornerWidth, 1);
    cornerHeight = std::max(cornerHeight, 1);

    // First corner position (top-left)
    std::vector<cv::Point2f> dstCorner = {
        {0.0f, 0.0f},
        {static_cast<float>(cornerWidth), 0.0f},
        {static_cast<float>(cornerWidth), static_cast<float>(cornerHeight)},
        {0.0f, static_cast<float>(cornerHeight)}
    };

    // Second corner position (bottom-right)
    float x0_opposite_corner = static_cast<float>(cardWidth  - cornerWidth);
    float y0_opposite_corner = static_cast<float>(cardHeight - cornerHeight);
    std::vector<cv::Point2f> dstOppositeCorner = {
        {x0_opposite_corner, y0_opposite_corner},
        {static_cast<float>(cardWidth), y0_opposite_corner},
        {static_cast<float>(cardWidth), static_cast<float>(cardHeight)},
        {x0_opposite_corner, static_cast<float>(cardHeight)}
    };

    // Transform back to original image space
    std::vector<cv::Point2f> srcCornerFloat, srcOppositeCornerFloat;
    cv::perspectiveTransform(dstCorner,srcCornerFloat, H_inv);
    cv::perspectiveTransform(dstOppositeCorner, srcOppositeCornerFloat, H_inv);

    // Convert cv::Point2f to cv::Point
    std::vector<cv::Point> srcCorner, srcOppositeCorner;
    convertToIntPoints(srcCornerFloat, srcCorner);
    convertToIntPoints(srcOppositeCornerFloat, srcOppositeCorner);

    bbox1 = cv::boundingRect(srcCorner);
    bbox2 = cv::boundingRect(srcOppositeCorner);

}

void convertToIntPoints(const std::vector<cv::Point2f>& foalPoints, std::vector<cv::Point>& intPoints) {
    intPoints.clear();
    intPoints.reserve(foalPoints.size());
    for (const auto& p : foalPoints) {
        intPoints.emplace_back(cvRound(p.x), cvRound(p.y));
    }
}


}

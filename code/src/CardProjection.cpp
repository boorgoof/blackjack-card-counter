#include "../include/CardProjection.h"
#include <opencv2/imgproc.hpp>

namespace CardProjection {

cv::Mat projectCard(const cv::Mat& image, const std::vector<cv::Point>& contour) {
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
    
    cv::Mat result;
    cv::warpPerspective(image, result, transform, cv::Size(250, 350));
    return result;
}

cv::Mat extractCardCorner(const cv::Mat& image, const std::vector<cv::Point>& contour, int width, int height) {
    cv::Mat fullCard = projectCard(image, contour);
    cv::Rect cornerRegion(0, 0, width, height);
    return fullCard(cornerRegion).clone();
}

}

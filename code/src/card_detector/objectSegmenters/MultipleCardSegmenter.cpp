#include "../../../include/card_detector/objectSegmenters/MultipleCardSegmenter.h"

MultipleCardSegmenter::MultipleCardSegmenter() {
    set_method_name("MultipleCard");
}

std::vector<std::vector<cv::Point>> MultipleCardSegmenter::segment_objects(
    const cv::Mat& src_img, 
    const cv::Mat& src_mask) {
    
    cv::Mat mask = src_mask.clone();
    
    if (mask.channels() > 1) {
        cv::cvtColor(mask, mask, cv::COLOR_BGR2GRAY);
    }
    cv::threshold(mask, mask, 127, 255, cv::THRESH_BINARY);
    
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, 
        cv::Size(params_.morphKernelSize, params_.morphKernelSize)
    );
    
    cv::Mat maskEroded;
    cv::erode(mask, maskEroded, kernel, cv::Point(-1, -1), params_.erosionIterations);
    
    cv::Mat distTransform;
    cv::distanceTransform(maskEroded, distTransform, cv::DIST_L2, 5);
    
    double minVal, maxVal;
    cv::minMaxLoc(distTransform, &minVal, &maxVal);
    
    if (maxVal < 1.0) {
        return {};
    }
    
    float threshold = static_cast<float>(maxVal) * params_.distThresholdPercent;
    std::vector<cv::Point> centers = findCardCenters(distTransform, threshold, params_.minDistanceBetweenCards);
    
    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
    
    std::vector<std::vector<cv::Point>> cardContours;
    cardContours.reserve(centers.size());
    
    for (const cv::Point& center : centers) {
        if (center.x < 0 || center.x >= labels.cols || center.y < 0 || center.y >= labels.rows) {
            continue;
        }
        
        int label = labels.at<int>(center);
        
        if (label == 0) {
            continue;
        }
        
        cv::Mat labelMask = (labels == label);
        labelMask.convertTo(labelMask, CV_8U);
        
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(labelMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        if (!contours.empty()) {
            std::vector<cv::Point> largestContour = contours[0];
            double maxArea = cv::contourArea(largestContour);
            
            for (size_t i = 1; i < contours.size(); i++) {
                double area = cv::contourArea(contours[i]);
                if (area > maxArea) {
                    maxArea = area;
                    largestContour = contours[i];
                }
            }
            
            if (maxArea >= params_.minCardArea) {
                cardContours.push_back(largestContour);
            }
        }
    }
    
    cv::Mat visualization = src_img.clone();
    for (size_t i = 0; i < cardContours.size(); i++) {
        cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);
        cv::drawContours(visualization, cardContours, i, color, 2);
    }
    cv::imshow("Multiple Card Segmentation", visualization);
    cv::waitKey(0);
    
    return cardContours;
}

std::vector<cv::Point> MultipleCardSegmenter::findCardCenters(
    const cv::Mat& dist, 
    float threshold, 
    int minDistance) {
    
    std::vector<cv::Point> centers;
    cv::Mat distCopy = dist.clone();
    
    while (true) {
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(distCopy, &minVal, &maxVal, &minLoc, &maxLoc);
        
        if (maxVal < threshold) {
            break;
        }
        
        bool tooClose = false;
        for (const cv::Point& existing : centers) {
            if (cv::norm(maxLoc - existing) < minDistance) {
                tooClose = true;
                break;
            }
        }
        
        if (!tooClose) {
            centers.push_back(maxLoc);
        }
        
        cv::circle(distCopy, maxLoc, minDistance / 2, cv::Scalar(0), -1);
        
        if (centers.size() >= static_cast<size_t>(params_.maxCards)) {
            break;
        }
    }
    
    return centers;
}

std::vector<cv::Point> MultipleCardSegmenter::findContourAroundCenter(
    const cv::Mat& mask, 
    const cv::Point& center, 
    int searchRadius) {
    
    cv::Rect roi(
        std::max(0, center.x - searchRadius),
        std::max(0, center.y - searchRadius),
        std::min(searchRadius * 2, mask.cols - std::max(0, center.x - searchRadius)),
        std::min(searchRadius * 2, mask.rows - std::max(0, center.y - searchRadius))
    );
    
    if (roi.width <= 0 || roi.height <= 0) {
        return {};
    }
    
    cv::Mat localMask = mask(roi).clone();
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(localMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    cv::Point localCenter(center.x - roi.x, center.y - roi.y);
    
    for (const std::vector<cv::Point>& contour : contours) {
        double test = cv::pointPolygonTest(contour, localCenter, false);
        if (test >= 0) {
            std::vector<cv::Point> globalContour;
            globalContour.reserve(contour.size());
            for (const cv::Point& pt : contour) {
                globalContour.push_back(cv::Point(pt.x + roi.x, pt.y + roi.y));
            }
            return globalContour;
        }
    }
    
    return {};
}

#include "../../../include/card_detector/objectSegmenters/DistanceTransformCardSegmenter.h"


DistanceTransformCardSegmenter::DistanceTransformCardSegmenter() {
    set_method_name("DistanceTransform");
}

std::vector<cv::RotatedRect> DistanceTransformCardSegmenter::segment_objects(
    const cv::Mat& src_img, 
    const cv::Mat& src_mask) {
    
    cv::Mat mask = src_mask.clone();
    
    // Ensure mask is binary
    if (mask.channels() > 1) {
        cv::cvtColor(mask, mask, cv::COLOR_BGR2GRAY);
    }
    cv::threshold(mask, mask, 127, 255, cv::THRESH_BINARY);
    
    // Compute distance transform
    cv::Mat distTransform;
    cv::distanceTransform(mask, distTransform, cv::DIST_L2, 5);
    
    // Find global max for threshold calculation
    double minVal, maxVal;
    cv::minMaxLoc(distTransform, &minVal, &maxVal);
    
    // Find local maxima (card centers)
    float threshold = static_cast<float>(maxVal) * params_.distThresholdPercent;
    std::vector<cv::Point> centers = findLocalMaxima(distTransform, threshold, params_.fixedMinDistance);
    
    // Extract rotated rectangles for each center
    std::vector<cv::RotatedRect> rotatedRects;
    rotatedRects.reserve(centers.size());
    
    for (const auto& center : centers) {
        cv::RotatedRect rotatedRect;
        
        // Get distance value at center
        float distValue = distTransform.at<float>(center);
        
        // Define ROI around the center
        int roiSize = std::max(30, static_cast<int>(distValue * params_.roiSizeMultiplier));
        
        cv::Rect roi(
            std::max(0, center.x - roiSize),
            std::max(0, center.y - roiSize),
            std::min(roiSize * 2, mask.cols - std::max(0, center.x - roiSize)),
            std::min(roiSize * 2, mask.rows - std::max(0, center.y - roiSize))
        );
        
        // Try to find local contour around this center
        bool foundLocalRect = false;
        if (roi.width > 30 && roi.height > 30) {
            cv::Mat localBinary = mask(roi).clone();
            std::vector<std::vector<cv::Point>> localContours;
            cv::findContours(localBinary, localContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            
            // Local center coordinates in ROI
            cv::Point localCenter(center.x - roi.x, center.y - roi.y);
            
            // Find contour that contains this center
            for (const auto& contour : localContours) {
                if (static_cast<int>(contour.size()) >= params_.minContourPoints) {
                    double test = cv::pointPolygonTest(contour, localCenter, false);
                    if (test >= 0) {
                        // Found the contour containing our center
                        cv::RotatedRect localRect = cv::minAreaRect(contour);
                        
                        // Transform back to global coordinates
                        localRect.center.x += roi.x;
                        localRect.center.y += roi.y;
                        
                        rotatedRect = localRect;
                        foundLocalRect = true;
                        break;
                    }
                }
            }
        }
        
        // Fallback: create rectangular region based on distance value
        if (!foundLocalRect) {
            float width = std::max(10.f, distValue * params_.bboxWidthMultiplier);
            float height = std::max(10.f, distValue * params_.bboxHeightMultiplier);
            rotatedRect = cv::RotatedRect(cv::Point2f(center), cv::Size2f(width, height), 0);
        }
        
        rotatedRects.push_back(rotatedRect);
    }
    
    return rotatedRects;
}

std::vector<cv::Point> DistanceTransformCardSegmenter::findLocalMaxima(
    const cv::Mat& dist, 
    float threshold, 
    int minDistance) {
    
    std::vector<cv::Point> maxima;
    cv::Mat distCopy = dist.clone();
    
    while (true) {
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(distCopy, &minVal, &maxVal, &minLoc, &maxLoc);
        
        // Stop if below threshold
        if (maxVal < threshold) break;
        
        // Check if too close to existing maxima
        bool tooClose = false;
        for (const auto& existing : maxima) {
            if (cv::norm(maxLoc - existing) < minDistance) {
                tooClose = true;
                break;
            }
        }
        
        // Add if not too close
        if (!tooClose) {
            maxima.push_back(maxLoc);
        }
        
        // Suppress this region
        cv::circle(distCopy, maxLoc, minDistance, cv::Scalar(0), -1);
        
        // Safety limit
        if (maxima.size() >= static_cast<size_t>(params_.maxCards)) break;
    }
    
    return maxima;
}
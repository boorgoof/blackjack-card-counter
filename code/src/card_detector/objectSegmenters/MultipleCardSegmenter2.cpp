#include "../../../include/card_detector/objectSegmenters/MultipleCardSegmenter2.h"
#include "../../../include/CardProjection.h"
#include "../../../include/card_detector/objectSegmenters/MultipleCardSegmenter.h"
//https://docs.opencv.org/3.4/d2/dbd/tutorial_distance_transform.html
std::vector<std::vector<cv::Point>> MultipleCardSegmenter2::segment_objects(const cv::Mat& image, const cv::Mat& mask) {
    
    cv::Mat maskBinary;
    if (mask.channels() > 1) {
        cv::cvtColor(mask, maskBinary, cv::COLOR_BGR2GRAY);
    } else {
        maskBinary = mask.clone();
    }
    cv::threshold(maskBinary, maskBinary, 127, 255, cv::THRESH_BINARY);
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(maskBinary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    cv::Mat visualization = image.clone();
    std::vector<std::vector<cv::Point>> results;
    
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area < 1000) continue;
        
        std::vector<cv::Point> approx;
        double epsilon = 0.005 * cv::arcLength(contours[i], true);
        cv::approxPolyDP(contours[i], approx, epsilon, true);
        
        // Merge lines with small angles
        std::vector<cv::Point> candidatePoints;
        double angleThreshold = 45.0; // degrees
        
        // First pass: collect all points with significant angles
        for (int j = 0; j < static_cast<int>(approx.size()); j++) {
            cv::Point p1 = approx[j];
            cv::Point p2 = approx[(j + 1) % approx.size()];
            cv::Point p3 = approx[(j + 2) % approx.size()];
            
            // Calculate vectors
            cv::Point v1 = p2 - p1;
            cv::Point v2 = p3 - p2;
            
            // Calculate angle between vectors
            double dot = v1.x * v2.x + v1.y * v2.y;
            double norm1 = sqrt(v1.x * v1.x + v1.y * v1.y);
            double norm2 = sqrt(v2.x * v2.x + v2.y * v2.y);
            
            double angle = 180.0;
            if (norm1 > 0 && norm2 > 0) {
                double cosAngle = dot / (norm1 * norm2);
                cosAngle = std::max(-1.0, std::min(1.0, cosAngle));
                angle = acos(cosAngle) * 180.0 / M_PI;
            }
            
            if (angle > angleThreshold) {
                candidatePoints.push_back(p2);
            }
        }
        
        // Second pass: filter points that are too close to each other
        std::vector<cv::Point> mergedApprox;
        double minDistance = 10.0; // pixels
        
        for (const cv::Point& candidate : candidatePoints) {
            bool tooClose = false;
            
            // Check distance with all already accepted points
            for (const cv::Point& accepted : mergedApprox) {
                double distance = sqrt(pow(candidate.x - accepted.x, 2) + pow(candidate.y - accepted.y, 2));
                if (distance < minDistance) {
                    tooClose = true;
                    std::cout << "Card " << i << " - Rejected point (" << candidate.x << "," << candidate.y 
                              << ") - too close to (" << accepted.x << "," << accepted.y << ") distance: " << distance << std::endl;
                    break;
                }
            }
            
            if (!tooClose) {
                mergedApprox.push_back(candidate);
                std::cout << "Card " << i << " - Accepted point (" << candidate.x << "," << candidate.y << ")" << std::endl;
            }
        }
        
        if (mergedApprox.size() < 3) {
            mergedApprox = approx; // Fallback to original if too few points
        }
        
        results.push_back(mergedApprox);
        
        // Test CardProjection with the detected points
        if (!mergedApprox.empty()) {
            cv::Mat projectedCard = CardProjection::projectCard(image, mergedApprox);
            
            // Create a white mask for the projected card
            cv::Mat projectedMask = cv::Mat::ones(projectedCard.size(), CV_8UC1) * 255;
            
            // Apply MultipleCardSegmenter to the projected card
            MultipleCardSegmenter segmenter;
            std::vector<std::vector<cv::Point>> segmentedCards = segmenter.segment_objects(projectedCard, projectedMask);
            
            cv::imshow("Original Image", image);
            cv::imshow("Projected Card " + std::to_string(i), projectedCard);
            cv::waitKey(0);
        }
        
        /*
        cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);
        
        for (int j = 0; j < static_cast<int>(mergedApprox.size()); j++) {
            cv::Point p1 = mergedApprox[j];
            cv::Point p2 = mergedApprox[(j + 1) % mergedApprox.size()];
            cv::line(visualization, p1, p2, color, 3);
        }
        
        // Draw angles on the image
        for (int j = 0; j < static_cast<int>(mergedApprox.size()); j++) {
            cv::Point p1 = mergedApprox[j];
            cv::Point p2 = mergedApprox[(j + 1) % mergedApprox.size()];
            cv::Point p3 = mergedApprox[(j + 2) % mergedApprox.size()];
            
            cv::Point v1 = p2 - p1;
            cv::Point v2 = p3 - p2;
            
            double dot = v1.x * v2.x + v1.y * v2.y;
            double norm1 = sqrt(v1.x * v1.x + v1.y * v1.y);
            double norm2 = sqrt(v2.x * v2.x + v2.y * v2.y);
            
            double angle = 180.0;
            if (norm1 > 0 && norm2 > 0) {
                double cosAngle = dot / (norm1 * norm2);
                cosAngle = std::max(-1.0, std::min(1.0, cosAngle));
                angle = acos(cosAngle) * 180.0 / M_PI;
            }
            
            if (angle > 45.0) {
                cv::circle(visualization, p2, 8, cv::Scalar(0, 255, 255), -1);
                cv::putText(visualization, std::to_string((int)angle) + "°", 
                           cv::Point(p2.x + 10, p2.y - 10), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
            }
        }
        
        cv::Rect bbox = cv::boundingRect(mergedApprox);
        cv::putText(visualization, "Card " + std::to_string(i) + " (" + std::to_string(mergedApprox.size()) + " lines)", 
                   cv::Point(bbox.x, bbox.y - 10), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        */
    }
    
    // cv::imshow("Bounding Box Lines", visualization);
    // cv::waitKey(0);
    
    return results;   
}
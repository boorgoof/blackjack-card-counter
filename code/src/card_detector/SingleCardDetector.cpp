#include "../../include/card_detector/SingleCardDetector.h"
#include <opencv2/imgproc.hpp>

SingleCardDetector::SingleCardDetector(RoughCardDetector* rough_card_detector, ObjectClassifier* object_classifier, bool detect_full_card, bool visualize)
    : CardDetector(detect_full_card, visualize), rough_card_detector_(rough_card_detector), object_classifier_(object_classifier) {
    
}

SingleCardDetector::~SingleCardDetector() {
    
}

cv::Mat SingleCardDetector::intersectRotatedRect(const cv::Mat& mask, const cv::RotatedRect& rect) const {
    // Ensure single-channel mask
    cv::Mat src = mask.clone();

    // Create a mask for the rotated rect
    cv::Mat rectMask(src.size(), CV_8UC1, cv::Scalar(0));
    cv::Point2f pt[4];
    rect.points(pt);
    std::vector<cv::Point> pts;
    pts.reserve(4);
    for (int i = 0; i < 4; ++i) {
        pts.emplace_back(cv::Point(cvRound(pt[i].x), cvRound(pt[i].y)));
    }
    cv::fillConvexPoly(rectMask, pts, cv::Scalar(255));

    // Bitwise AND to keep only pixels inside
    cv::Mat out;
    cv::bitwise_and(src, rectMask, out);
    return out;
}

std::vector<Label> SingleCardDetector::detect_image(const cv::Mat& image) {
    std::vector<Label> detected_labels;

    //first, use the rough card detector to get a mask of the area where the card is located
    cv::Mat mask = this->rough_card_detector_->getMask(image);

    cv::RotatedRect card_rect;
    
    cv::Mat mask2 = this->intersectRotatedRect(mask, card_rect);


    if (this->object_classifier_) {
        this->object_classifier_->classify_object(image, mask2);
    }

    return detected_labels;
}


#include "../../include/card_detector/SingleCardDetector.h"
#include <opencv2/imgproc.hpp>

SingleCardDetector::SingleCardDetector(RoughCardDetector* rough_card_detector, ObjectClassifier* object_classifier, ObjectSegmenter* object_segmenter, bool detect_full_card, bool visualize)
    : CardDetector(detect_full_card, visualize), rough_card_detector_(rough_card_detector), object_classifier_(object_classifier), object_segmenter_(object_segmenter) {
    
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

    cv::Mat mask = this->rough_card_detector_->getMask(image);
    std::vector<cv::RotatedRect> cards_rect = this->object_segmenter_->segment_objects(image, mask);

    for (const auto& rotated_rect : cards_rect) {
        cv::Mat single_obj_mask = this->intersectRotatedRect(mask, rotated_rect);

        if (this->object_classifier_) {
            const ObjectType* obj_type = this->object_classifier_->classify_object(image, single_obj_mask);
            
            if (obj_type && obj_type->isValid()) {

                cv::Rect bounding_box = rotated_rect.boundingRect();
                Label label(obj_type->clone(), bounding_box);
                detected_labels.push_back(std::move(label));
                
            }
        }
    }

    return detected_labels;
}

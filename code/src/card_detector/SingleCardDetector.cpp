#include "../../include/card_detector/SingleCardDetector.h"
#include "../../include/CardProjection.h"
#include <opencv2/imgproc.hpp>

SingleCardDetector::SingleCardDetector(std::unique_ptr<RoughCardDetector> rough_card_detector, std::unique_ptr<ObjectClassifier> object_classifier, std::unique_ptr<ObjectSegmenter> object_segmenter, const bool detect_full_card, const bool visualize)
    : CardDetector(detect_full_card, visualize), rough_card_detector_(std::move(rough_card_detector)), object_classifier_(std::move(object_classifier)), object_segmenter_(std::move(object_segmenter)) {

}

SingleCardDetector::~SingleCardDetector() {
    
}
//todo delete
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



cv::Mat SingleCardDetector::intersectContour(const cv::Mat& mask, const std::vector<cv::Point>& contour) const {
    // Ensure single-channel mask
    cv::Mat src = mask.clone();

    // Create a mask for the contour
    cv::Mat contourMask(src.size(), CV_8UC1, cv::Scalar(0));
    
    // Fill the contour on the mask
    std::vector<std::vector<cv::Point>> contours = {contour};
    cv::fillPoly(contourMask, contours, cv::Scalar(255));

    // Bitwise AND to keep only pixels inside
    cv::Mat out;
    cv::bitwise_and(src, contourMask, out);
    return out;
}

std::vector<Label> SingleCardDetector::detect_image(const cv::Mat& image) {
    std::vector<Label> detected_labels;

    cv::Mat mask = this->rough_card_detector_->getMask(image);
    std::vector<std::vector<cv::Point>> cards_contour = this->object_segmenter_->segment_objects(image, mask);
    
    for (const std::vector<cv::Point>& contour : cards_contour) {
        cv::Mat single_obj_mask = this->intersectContour(mask, contour);
        //Fede lavora qua con projected, prova a usare qualche filtro per migliorare la classificazione
        cv::Mat projected = CardProjection::projectCard(image, contour);
        cv::Mat corner = CardProjection::extractCardCorner(image, contour);

        if (this->object_classifier_) {
            const ObjectType* obj_type = this->object_classifier_->classify_object(image, single_obj_mask);
            
            if (obj_type && obj_type->isValid()) {

                cv::Rect bounding_box = boundingRect(contour);
                Label label(obj_type->clone(), bounding_box);
                detected_labels.push_back(std::move(label));
                
            }
        }
    }

    return detected_labels;
}
#include "../../include/card_detector/SingleCardDetector.h"
#include "../../include/card_detector/CardProjection.h"
#include "../../include/ObjectType.h"
#include <opencv2/imgproc.hpp>

SingleCardDetector::SingleCardDetector(RoughCardDetector* rough_card_detector, ObjectClassifier* object_classifier, ObjectSegmenter* object_segmenter, bool detect_full_card, bool visualize)
    : CardDetector(detect_full_card, visualize), rough_card_detector_(rough_card_detector), object_classifier_(object_classifier), object_segmenter_(object_segmenter) {
    
}

SingleCardDetector::~SingleCardDetector() {
    
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
   

    for (std::vector<cv::Point>& contour : cards_contour) {
        
        cv::Mat single_obj_mask; 
        cv::Mat card_projected_image;
         cv::Rect bbox1, bbox2;
        
        if(this->object_segmenter_->get_method_name()== "SimpleContours"){
            
            // for the test dataset we need to find the rectangles of a card 
            cv::Mat H = CardProjection::getPerspectiveTranform(image, contour);
            cv::warpPerspective(image, card_projected_image, H, cv::Size(250, 350));
            cv::Mat H_inv = H.inv();

            const int cardWidth = card_projected_image.cols;
            const int cardHeight = card_projected_image.rows; 
            
            CardProjection::compute_two_opposite_corners_bboxes(H_inv, cardWidth, cardHeight, bbox1, bbox2);

        }else{
            single_obj_mask = this->intersectContour(mask, contour);
        }

        
        if (this->object_classifier_) {

            const ObjectType* obj_type = nullptr;

            if (!card_projected_image.empty()) {
                obj_type = this->object_classifier_->classify_object(card_projected_image, cv::Mat());
            } else {
                obj_type = this->object_classifier_->classify_object(image, single_obj_mask);
            }
            
            if (obj_type && obj_type->isValid()) {

                std::vector<cv::Rect> bboxes;
                if (bbox1.empty() == false && bbox2.empty() == false) {
                    bboxes.push_back(bbox1);
                    bboxes.push_back(bbox2);
                    Label label(obj_type->clone(), bboxes);  
                    detected_labels.push_back(std::move(label));
                } else {
                    cv::Rect bounding_box = boundingRect(contour);
                    Label label(obj_type->clone(), bounding_box);
                    detected_labels.push_back(std::move(label));
                }
                   
            }
        }
    }

    return detected_labels;
}


#include "../../../include/detection/card_detector/CardProjection.h"
#include "../../../include/detection/card_detector/SegmentationClassificationCardDetector.h"
#include "../../../include/ObjectType.h"
#include "../../../include/ImageFilter.h"
#include <opencv2/imgproc.hpp>

SegmentationClassificationCardDetector::SegmentationClassificationCardDetector(std::unique_ptr<MaskCardDetector> mask_card_detector,std::unique_ptr<ObjectClassifier> object_classifier, std::unique_ptr<ObjectSegmenter> object_segmenter)
    : CardDetector(), mask_card_detector_(std::move(mask_card_detector)), object_classifier_(std::move(object_classifier)), object_segmenter_(std::move(object_segmenter)) {
        
}

std::vector<Label> SegmentationClassificationCardDetector::detect_cards(const cv::Mat& image) {

    std::vector<Label> detected_labels;

    cv::Mat mask = this->mask_card_detector_->getMask(image);
    std::vector<std::vector<cv::Point>> cards_contour = this->object_segmenter_->segment_objects(image, mask);
   
    for (std::vector<cv::Point>& contour : cards_contour) {
        
        // we find the bounding boxes of the two corners of the card
        cv::Mat card_projected_image;
        cv::Rect bbox1, bbox2;
        
        cv::Mat H = CardProjection::getPerspectiveTranform(image, contour);
        cv::warpPerspective(image, card_projected_image, H, cv::Size(250, 350));
        cv::Mat H_inv = H.inv();

        const int cardWidth = card_projected_image.cols;
        const int cardHeight = card_projected_image.rows; 
        CardProjection::compute_two_opposite_corners_bboxes(H_inv, cardWidth, cardHeight, bbox1, bbox2);


        if (this->object_classifier_) {

            const ObjectType* obj_type = nullptr;
            
            card_projected_image = Filters::unsharp_mask(card_projected_image, 1.6, 1.5);
            card_projected_image = Filters::CLAHE_contrast_equalization(card_projected_image, 2, 8);
            cv::imshow("Projected Card", card_projected_image);
            cv::waitKey(0);
            obj_type = this->object_classifier_->classify_object(card_projected_image, cv::Mat());
            

            if (obj_type && obj_type->isValid()) {

                std::vector<cv::Rect> bboxes;
                bboxes.push_back(bbox1);
                bboxes.push_back(bbox2);
                Label label(obj_type->clone(), bboxes);  
                detected_labels.push_back(std::move(label));     
            }
        }
    }

    return detected_labels;
}


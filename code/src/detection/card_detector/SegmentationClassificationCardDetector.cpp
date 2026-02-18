#include "../../../include/detection/card_detector/CardProjection.h"
#include "../../../include/detection/card_detector/SegmentationClassificationCardDetector.h"
#include "../../../include/ObjectType.h"
#include "../../../include/ImageFilter.h"
#include <opencv2/imgproc.hpp>

SegmentationClassificationCardDetector::SegmentationClassificationCardDetector(std::unique_ptr<MaskCardDetector> mask_card_detector,std::unique_ptr<ObjectClassifier> object_classifier, std::unique_ptr<ObjectSegmenter> object_segmenter, bool visualize)
    : CardDetector(visualize), mask_card_detector_(std::move(mask_card_detector)), object_classifier_(std::move(object_classifier)), object_segmenter_(std::move(object_segmenter)) {
        
}

std::vector<Label> SegmentationClassificationCardDetector::detect_cards(const cv::Mat& image) {

    std::vector<Label> detected_labels;
    // small preprocessing 
    cv::Mat filtered_image = Filters::delete_gray_background(image);
    
    // get the mask 
    cv::Mat mask = this->mask_card_detector_->getMask(filtered_image);
    cv::Mat white_masked_image(image.size(), image.type(), cv::Scalar(255,255,255));
    cv::Mat black_masked_image(image.size(), image.type(), cv::Scalar(0,0,0));
    image.copyTo(white_masked_image, mask);   
    image.copyTo(black_masked_image, mask);

    // segment the card using the mask
    std::vector<std::vector<cv::Point>> cards_contour = this->object_segmenter_->segment_objects(black_masked_image, mask); 

    for (std::vector<cv::Point>& contour : cards_contour) {
        
        // Get projected card and corner bounding boxes in original image coordinates
        cv::Mat card_projected_image;
        cv::Rect bbox1, bbox2;
        CardProjection::getCornerBboxes(white_masked_image, contour, bbox1, bbox2, card_projected_image);

        // now we classify the card using the projected image
        if (this->object_classifier_) {

            const ObjectType* obj_type = nullptr;
            
            // This is an optional step to try to improve the classification by binarizing the projected card image 
            //card_color_utils::CardColor color = detect_card_color(card_projected_image);
            //card_projected_image = Filters::two_color_binarization(card_projected_image, card_color_utils::to_scalar(color), cv::Scalar(255,255,255));
            
            // classify the card using the projected image
            obj_type = this->object_classifier_->classify_object(card_projected_image, cv::Mat());
            
            // create label only if the classification is valid
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

card_color_utils::CardColor SegmentationClassificationCardDetector::detect_card_color(const cv::Mat& card_img)
{
    if (card_img.empty()) {
        return card_color_utils::CardColor::UNKNOWN;
    }

    // we convert the image to HSV 
    cv::Mat hsv;
    cv::cvtColor(card_img, hsv, cv::COLOR_BGR2HSV);

    // we want to use only the colored pixels (red), not the black/white ones
    const int satMin = 50;
    cv::Mat satMask;
    cv::inRange(hsv, cv::Scalar(0, satMin, 0), cv::Scalar(180, 255, 255), satMask);

    // for the color detection we use the channel H 
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    cv::Mat H = channels[0];

    int histSize = 180;
    float range[] = {0.f, 180.f};
    const float* histRange[] = { range };
    int ch[] = {0};

    cv::Mat hist;
    cv::calcHist(&H, 1, ch, satMask, hist, 1, &histSize, histRange, true, false);
    
    // we check each bin of the histogram to find the maximum
    int maxBin = 0;
    double maxVal = 0.0;
    for (int i = 0; i < histSize; ++i) {
        float v = hist.at<float>(i);
        if (v > maxVal) {
            maxVal = v;
            maxBin = i;
        }
    }

    if (maxVal <= 0.0) {
        return card_color_utils::CardColor::BLACK;
    }

    // we check if the peak is red (otherwise is black)
    bool peakIsRed = (maxBin <= 10) || (maxBin >= 170);
    if (peakIsRed) {
        return card_color_utils::CardColor::RED;
    } else {
        return card_color_utils::CardColor::BLACK;
    }
    
}


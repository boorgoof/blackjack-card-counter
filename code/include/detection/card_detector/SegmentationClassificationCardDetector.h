// Federico Meneghetti

#ifndef SEGMENTATION_CLASSIFICATION_CARD_DETECTOR_H
#define SEGMENTATION_CLASSIFICATION_CARD_DETECTOR_H

#include "CardDetector.h"
#include "../../CardType.h"
#include "MaskCardDetector.h"
#include "objectClassifiers/ObjectClassifier.h"
#include "objectSegmenters/ObjectSegmenter.h"


class SegmentationClassificationCardDetector : public CardDetector {
public:

    /**
     * @brief constructor for the SegmentationClassificationCardDetector class
     * @param mask_card_detector a unique pointer to a MaskCardDetector object that will be used to generate masks for card detection
     * @param object_classifier a unique pointer to an ObjectClassifier object that will be used to classify the detected cards
     * @param object_segmenter a unique pointer to an ObjectSegmenter object that will be used to segment the detected cards
     * @param visualize a boolean to indicate whether to visualize the detected cards or not 
     */
    SegmentationClassificationCardDetector(std::unique_ptr<MaskCardDetector> mask_card_detector, std::unique_ptr<ObjectClassifier> object_classifier, std::unique_ptr<ObjectSegmenter> object_segmenter, bool visualize);
    ~SegmentationClassificationCardDetector() override = default;

    std::vector<Label> detect_cards(const cv::Mat& image) override;
    card_color_utils::CardColor detect_card_color(const cv::Mat& card_img);


private:
    std::unique_ptr<MaskCardDetector> mask_card_detector_;
    std::unique_ptr<ObjectClassifier> object_classifier_;
    std::unique_ptr<ObjectSegmenter> object_segmenter_;
};

#endif // SEG_CLASS_CARD_DETECTOR_H
#ifndef SEGMENTATION_CLASSIFICATION_CARD_DETECTOR_H
#define SEGMENTATION_CLASSIFICATION_CARD_DETECTOR_H

#include "CardDetector.h"
#include "MaskCardDetector.h"
#include "objectClassifiers/ObjectClassifier.h"
#include "objectSegmenters/ObjectSegmenter.h"


class SegmentationClassificationCardDetector : public CardDetector {
public:

    SegmentationClassificationCardDetector(std::unique_ptr<MaskCardDetector> mask_card_detector, std::unique_ptr<ObjectClassifier> object_classifier, std::unique_ptr<ObjectSegmenter> object_segmenter);
    ~SegmentationClassificationCardDetector() override = default;

    std::vector<Label> detect_cards(const cv::Mat& image) override;

private:
    std::unique_ptr<MaskCardDetector> mask_card_detector_;
    std::unique_ptr<ObjectClassifier> object_classifier_;
    std::unique_ptr<ObjectSegmenter> object_segmenter_;
};

#endif // SEG_CLASS_CARD_DETECTOR_H
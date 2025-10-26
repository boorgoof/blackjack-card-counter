#ifndef SINGLE_CARD_DETECTOR_H
#define SINGLE_CARD_DETECTOR_H

#include "CardDetector.h"
#include "RoughCardDetector.h"
#include "objectClassifiers/ObjectClassifier.h"
#include "objectSegmenters/ObjectSegmenter.h"

class SingleCardDetector : public CardDetector {
public:
    SingleCardDetector(RoughCardDetector* rough_card_detector, ObjectClassifier* object_classifier, ObjectSegmenter* object_segmenter, bool detect_full_card = false, bool visualize = false);
    ~SingleCardDetector();

    std::vector<Label> detect_image(const cv::Mat& image) override;
    
private:
    cv::Mat intersectRotatedRect(const cv::Mat& mask, const cv::RotatedRect& rect) const;
    std::unique_ptr<RoughCardDetector> rough_card_detector_;
    std::unique_ptr<ObjectClassifier> object_classifier_;
    std::unique_ptr<ObjectSegmenter> object_segmenter_;

};

#endif
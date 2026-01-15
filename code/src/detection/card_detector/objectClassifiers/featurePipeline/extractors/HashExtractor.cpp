#include "../../../../../../include/detection/card_detector/objectClassifiers/featurePipeline/extractors/HashExtractor.h"
#include "../../../../../../include/Utils.h"
void HashExtractor::init()
{
    switch(this->type_) {
        case ExtractorType::FeatureDescriptorAlgorithm::PHASH:
            this->hasher_ = cv::img_hash::PHash::create();
            break;
        case ExtractorType::FeatureDescriptorAlgorithm::COLOR_MOMENT_HASH:
            this->hasher_ = cv::img_hash::ColorMomentHash::create();
            break;
        default:
            throw std::invalid_argument("Invalid HashExtractor type");
    }
}

Feature * HashExtractor::extractFeatures(const cv::Mat & img, const cv::Mat & mask) const
{
    cv::Mat hash;
    this->hasher_->compute(img, hash);
    return new HashFeature(hash);
}
#ifndef FEATUREPIPELINE_H
#define FEATUREPIPELINE_H

#include "../ObjectClassifier.h"
#include "FeatureExtractor.h"
#include "FeatureMatcher.h"

#include "../../../../Label.h"
#include "features/FeatureContainer.h"
#include "../../../../Dataset/TemplateDataset.h"
#include <opencv2/opencv.hpp>



class FeaturePipeline : public ObjectClassifier {

private:
    /**
    * @brief FeatureExtractor pointer to the feature extractor used by the pipeline.
    */
    std::unique_ptr<FeatureExtractor> extractor_;
    /**
     * @brief FeatureMatcher pointer to the feature matcher used by the pipeline.
     */
    std::unique_ptr<FeatureMatcher> matcher_;

    const std::map<const ObjectType*, const Feature*>& template_features_;

    /**
     * @brief  check and to update the compatibility between the extractor and matcher.
     */
    void update_extractor_matcher_compatibility();

    float computeConfidence(size_t num_inliers, size_t total_matches) const;

public:

    FeaturePipeline(FeatureExtractor* extractor, FeatureMatcher* matcher, TemplateDataset& template_dataset);

    FeaturePipeline(const ExtractorType::FeatureDescriptorAlgorithm extractor, const MatcherType::MatcherAlgorithm matcher, TemplateDataset& template_dataset);

    ~FeaturePipeline();

    
    const ObjectType* classify_object(const cv::Mat& src_img,  const cv::Mat &src_mask) override;

    void setExtractororComponent(FeatureExtractor* fd) {
        this->extractor_.reset(fd);
    }
    void setMatcherComponent(FeatureMatcher* fm) {
        this->matcher_.reset(fm);
    }
    
};


#endif // FEATUREPIPELINE_H
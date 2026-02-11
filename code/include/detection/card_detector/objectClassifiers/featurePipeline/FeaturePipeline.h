#ifndef FEATUREPIPELINE_H
#define FEATUREPIPELINE_H

#include "../ObjectClassifier.h"
#include "extractors/FeatureExtractor.h"
#include "matchers/FeatureMatcher.h"

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

    const std::map<const ObjectType*, const Feature*>* template_features_;

    double ratio_threshold_ = 0.8;
    
public:

    FeaturePipeline(ExtractorType::FeatureDescriptorAlgorithm extractor_agorithm, const double lowe_ratio_threshold, TemplateDataset& template_dataset);

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
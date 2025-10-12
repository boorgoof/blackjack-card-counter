#ifndef FEATUREPIPELINE_H
#define FEATUREPIPELINE_H

#include "../ObjectDetector.h"
#include "FeatureExtractor.h"
#include "FeatureMatcher.h"


#include "../../../Label.h"
#include "features/FeatureContainer.h"

#include <opencv2/opencv.hpp>



/**
 * @brief FeaturePipeline class to detect objects in images using feature extraction and matching.
 *        The class derives from the abstract class ObjectDetector.
 */
class FeaturePipeline : public ObjectDetector {

private:
    /**
    * @brief FeatureExtractor pointer to the feature extractor used by the pipeline.
    */
    std::unique_ptr<FeatureExtractor> extractor_;
    /**
     * @brief FeatureMatcher pointer to the feature matcher used by the pipeline.
     */
    std::unique_ptr<FeatureMatcher> matcher_;

    const std::map<ObjectType*, Feature*>& template_features_;
    /**
     * @brief  check and to update the compatibility between the extractor and matcher.
     */
    void update_extractor_matcher_compatibility();

    bool findBoundingBox(const std::vector<cv::DMatch>& matches,
                                    const KeypointFeature& templFeatures,       
                                    const KeypointFeature& imgFeatures,        
                                    Label& out_label, 
                                    std::vector<unsigned char>& out_inlier_mask) const;

    void nmsLabels(std::vector<Label>& labels, double iou_thresh) const;

public:

    FeaturePipeline(FeatureExtractor* extractor, FeatureMatcher* matcher, const std::string& templates_folder_path);

    FeaturePipeline(const ExtractorType::FeatureDescriptorAlgorithm extractor, const MatcherType::MatcherAlgorithm matcher, const std::string& templates_folder_path);
    ~FeaturePipeline();

    void setExtractororComponent(FeatureExtractor* fd) {
        this->extractor_.reset(fd);
    }
    void setMatcherComponent(FeatureMatcher* fm) {
        this->matcher_.reset(fm);
    }

    void detect_objects(const cv::Mat& src_img, const cv::Mat &src_mask, std::vector<Label>& out_labels) override;

              
};


#endif // FEATUREPIPELINE_H
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

    size_t minMatchesForRANSAC_;   // matches needed to apply RANSAC
    size_t numMinInliers_;   // min inliers to validate the found bbox
    double nmsIoU_; // non-maxima suppression IoU threshold
    double numRansacReprojErr_;  // RANSAC reprojection error
    size_t numMaxInstancesPerTemplate_;    // max instances per template

    bool findBoundingBox(const std::vector<cv::DMatch>& matches,
                                    const KeypointFeature& templFeatures,       
                                    const KeypointFeature& imgFeatures,        
                                    Label& out_label, 
                                    std::vector<unsigned char>& out_inlier_mask) const;

    void nmsLabels(std::vector<Label>& labels, double iou_thresh) const;

public:

    FeaturePipeline(FeatureExtractor* extractor, FeatureMatcher* matcher, const std::string& templates_folder_path, 
                    size_t minMatchesForRANSAC = 50,
                    size_t numMinInliers = 20,
                    double nmsIoU = 0.30,
                    double numRansacReprojErr = 3.0,
                    size_t numMaxInstancesPerTemplate = 5);

    FeaturePipeline(const ExtractorType::FeatureDescriptorAlgorithm extractor, const MatcherType::MatcherAlgorithm matcher, const std::string& templates_folder_path,
                    size_t minMatchesForRANSAC = 50,
                    size_t numMinInliers = 20,
                    double nmsIoU = 0.30,
                    double numRansacReprojErr = 3.0,
                    size_t numMaxInstancesPerTemplate = 5);

    ~FeaturePipeline();

    
    void detect_objects(const cv::Mat& src_img, const cv::Mat &src_mask, std::vector<Label>& out_labels) override;

    void setExtractororComponent(FeatureExtractor* fd) {
        this->extractor_.reset(fd);
    }
    void setMatcherComponent(FeatureMatcher* fm) {
        this->matcher_.reset(fm);
    }
    void setMinMatchesForRANSAC(size_t v) { minMatchesForRANSAC_ = v; }
    void setNumMinInliers(size_t v) { numMinInliers_ = v; }
    void setNmsIoU(double v){ nmsIoU_ = v; }
    void setNumRansacReprojErr(double v) { numRansacReprojErr_ = v; }
    void setNumMaxInstancesPerTemplate(size_t v){ numMaxInstancesPerTemplate_ = v; }

    // Getter
    size_t minMatchesForRANSAC()  const { return minMatchesForRANSAC_; }
    size_t numMinInliers() const { return numMinInliers_; }
    double nmsIoU() const { return nmsIoU_; }
    double numRansacReprojErr() const { return numRansacReprojErr_; }
    size_t numMaxInstancesPerTemplate() const { return numMaxInstancesPerTemplate_; }
          
};


#endif // FEATUREPIPELINE_H
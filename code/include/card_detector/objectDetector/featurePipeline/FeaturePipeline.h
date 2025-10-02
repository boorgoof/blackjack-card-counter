#ifndef FEATUREPIPELINE_H
#define FEATUREPIPELINE_H

#include "../ObjectDetector.h"
#include "FeatureExtractor.h"
#include "FeatureMatcher.h"
#include "Features.h"

#include "../../../Label.h"
#include "../../../FeatureContainer.h"
#include "../../../FeatureDescriptorAlgorithm.h"

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
        std::unique_ptr<FeatureExtractor> extractor;
        /**
         * @brief FeatureMatcher pointer to the feature matcher used by the pipeline.
         */
        std::unique_ptr<FeatureMatcher> matcher;
    
        const std::map<Card_Type, cv::InputArray>& template_descriptors;

        /**
         * @brief  check and to update the compatibility between the extractor and matcher.
         */
        void update_extractor_matcher_compatibility();


        /**
        * @brief find the label (so the bounding box) of the object in the scene image.
        * @param matches the matches between the model and the scene
        * @param model_keypoint the keypoints of the model
        * @param scene_keypoint the keypoints of the scene (test image)
        * @param img_model the model image
        * @param mask_model the model mask
        * @param img_scene the scene (test) image
        * @param object_type the object type to find
        * @return the Label that containt the bounding box of the object in the scene
        */
        Label findBoundingBox(const std::vector<cv::DMatch>& matches,
            const std::vector<cv::KeyPoint>& model_keypoint,
            const std::vector<cv::KeyPoint>& scene_keypoint,
            const cv::Mat& img_model,
            const cv::Mat& mask_model,
            const cv::Mat& img_scene,
            Card_Type object_type) const ;

    public:

        /**
         * @brief This constructor initialize member variables, checks the compatibility between the extractor and matcher,
         *        and sets the method's name and the filter's names.
         * 
         * @param extractor pointer to the feature extractor used by the pipeline.
         * @param matcher pointer to the feature matcher used by the pipeline.
         */
        FeaturePipeline(FeatureExtractor* extractor, FeatureMatcher* matcher, const FeatureDescriptorAlgorithm& algoDescriptor);
        ~FeaturePipeline();

        void setExtractororComponent(FeatureExtractor* fd) {
            this->extractor.reset(fd);
        }
        void setMatcherComponent(FeatureMatcher* fm) {
            this->matcher.reset(fm);
        }


        /**
         * @brief detect objects in the scene image.
         * @param src_img the scene (test) image
         * @param out_labels the output vector of labels that will contain the detected objects
         */
        void detect_objects(const cv::Mat& src_img, std::vector<Label>& out_labels) override;

              
};

#endif // FEATUREPIPELINE_H
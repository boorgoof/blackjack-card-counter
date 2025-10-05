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
        std::unique_ptr<FeatureExtractor> extractor;
        /**
         * @brief FeatureMatcher pointer to the feature matcher used by the pipeline.
         */
        std::unique_ptr<FeatureMatcher> matcher;
    
        std::shared_ptr<const std::map<Card_Type, Feature*>> template_features; 
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
        bool FeaturePipeline::findBoundingBox(const std::vector<cv::DMatch>& matches,
                                      const KeypointFeature* templFeatures,       
                                      const KeypointFeature& imgFeatures,        
                                      Card_Type card_template,
                                      Label& out_label, // todo modify to objtype maybe do a virtual class
                                      std::vector<unsigned char>& out_inlier_mask) const;

    public:

        /**
         * @brief This constructor initialize member variables, checks the compatibility between the extractor and matcher,
         *        and sets the method's name and the filter's names.
         * 
         * @param extractor pointer to the feature extractor used by the pipeline.
         * @param matcher pointer to the feature matcher used by the pipeline.
         */
        FeaturePipeline(FeatureExtractor* extractor, FeatureMatcher* matcher, const std::string& template_cards_folder_path);

        /**
         * @brief This constructor initialize member variables, checks the compatibility between the extractor and matcher,
         *        and sets the method's name and the filter's names.
         * 
         * @param extractor 
         * @param matcher
         */
        FeaturePipeline(ExtractorType::FeatureDescriptorAlgorithm extractor, MatcherType::MatcherAlgorithm matcher, const std::string& template_cards_folder_path);
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
        void detect_objects(const cv::Mat& src_img, const cv::Mat &src_mask, std::vector<Label>& out_labels) override;

              
};


#endif // FEATUREPIPELINE_H
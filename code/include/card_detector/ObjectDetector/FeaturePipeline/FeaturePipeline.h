//Federico Meneghetti

#ifndef FEATUREPIPELINE_H
#define FEATUREPIPELINE_H

#include "../ObjectDetector.h"
#include "FeatureDetector.h"
#include "FeatureMatcher.h"
#include <opencv2/opencv.hpp>
#include "../../Label.h"
#include "Features.h"
#include "../../ImageFilter.h"

/**
 * @brief FeaturePipeline class to detect objects in images using feature detection and matching.
 *        The class derives from the abstract class ObjectDetector.
 */
class FeaturePipeline : public ObjectDetector {

    private:
        /**
        * @brief FeatureDetector pointer to the feature detector used by the pipeline.
        */
        std::unique_ptr<FeatureDetector> detector;
        /**
         * @brief FeatureMatcher pointer to the feature matcher used by the pipeline.
         */
        std::unique_ptr<FeatureMatcher> matcher;
    
        /**
         * @brief ImageFilter pointer to the image filter used by the pipeline.
         */
        std::unique_ptr<ImageFilter> model_imagefilter;
        /**
         * @brief ImageFilter pointer to the image filter used by the pipeline.
         */
        std::unique_ptr<ImageFilter> test_imagefilter;

        /**
         * @brief Dataset reference to the dataset used by the pipeline.
         */
        Dataset& dataset;

        /**
         * @brief vector of ModelFeatures to store the features of the all the dataset's models.
         */
        std::vector<ModelFeatures> models_features;

         /**
         * @brief initialize all the models' features.
         */
        void init_models_features();

        /**
         * @brief  check and to update the compatibility between the detector and matcher.
         */
        void update_detector_matcher_compatibility();

    public:

        /**
         * @brief This constructor initialize member variables, 
         *        checks the compatibility between the detector and matcher, 
         *        calculates all the models' features
         *        and sets the method's name and the filter's names.
         * @param detector pointer to the feature detector used by the pipeline.
         * @param matcher pointer to the feature matcher used by the pipeline.
         * @param dataset reference to the dataset used by the pipeline.
         * @param model_imagefilter pointer to the image filter used by the pipeline.
         * @param test_imagefilter pointer to the image filter used by the pipeline.
         */
        FeaturePipeline(FeatureDetector* detector, FeatureMatcher* matcher, Dataset& dataset, std::unique_ptr<ImageFilter> model_imagefilter = nullptr, std::unique_ptr<ImageFilter> test_imagefilter = nullptr);
        ~FeaturePipeline();

        void addDetectorComponent(FeatureDetector* fd) {
            this->detector.reset(fd);
        }
        void addMatcherComponent(FeatureMatcher* fm) {
            this->matcher.reset(fm);
        }
        void addModelImageFilterComponent(std::unique_ptr<ImageFilter> imagefilter) {
            this->model_imagefilter = std::move(imagefilter);
        }
        void addTestImageFilterComponent(std::unique_ptr<ImageFilter> imagefilter) {
            this->test_imagefilter = std::move(imagefilter);
        }
        void setDataset(Dataset& dataset) {
            this->dataset = dataset;
        }
        const Dataset& getDataset() const {
            return this->dataset;
        }
        
        /**
         * @brief detect objects in the scene image.
         * @param src_img the scene (test) image
         * @param out_labels the output vector of labels that will contain the detected objects
         */
        void detect_objects(const cv::Mat& src_img, std::vector<Label>& out_labels) override;

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
            Object_Type object_type) const ;
        
       
};

#endif // FEATUREPIPELINE_H



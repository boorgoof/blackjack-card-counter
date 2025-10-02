#include "../../../../include/card_detector/ObjectDetector/FeaturePipeline/FeaturePipeline.h"


void FeaturePipeline::update_extractor_matcher_compatibility() {
    if (this->extractor->getType() == ExtractorType::Type::ORB && this->matcher->getType() == MatcherType::Type::FLANN) {
        this->matcher = std::make_unique<FeatureMatcher>(FeatureMatcher(MatcherType::Type::FLANN, new cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2))));
    }
}

FeaturePipeline::~FeaturePipeline() {}


FeaturePipeline::FeaturePipeline(FeatureExtractor* extractor, FeatureMatcher* matcher, const FeatureDescriptorAlgorithm& algoDescriptor)
    : extractor{extractor}, matcher{matcher} {

    this->update_extractor_matcher_compatibility();

    this->template_descriptors = FeatureContainerFor<FeatureDescriptorAlgorithm::SIFT>::getInstance().get();
    
    /*
    switch (algoDescriptor)
    {
        case FeatureDescriptorAlgorithm::SIFT:
            this->template_descriptors = FeatureContainerFor<FeatureDescriptorAlgorithm::SIFT>::getInstance().get();
            break;

        case FeatureDescriptorAlgorithm::ORB:
            this->template_descriptors = FeatureContainerFor<FeatureDescriptorAlgorithm::ORB>::getInstance().get();
            break;

        case FeatureDescriptorAlgorithm::HASH:
            this->template_descriptors = FeatureContainerFor<FeatureDescriptorAlgorithm::HASH>::getInstance().get();
            break;
    }*/

    std::string method_name = ExtractorType::toString(extractor->getType()) + "-" + MatcherType::toString(matcher->getType());
    this->set_method_name(method_name);
}

void FeaturePipeline::detect_objects(const cv::Mat &src_img, std::vector<Label> &out_labels) {

    out_labels.clear();

    //models' features are already detected and stored in the pipeline (they always remain the same for every test image, so they are detected only once)

    cv::Mat src_img_filtered = src_img.clone();
    //test image filtering if the filter component is present
    if (this->test_imagefilter != nullptr) {
        src_img_filtered = this->test_imagefilter->apply_filters(src_img);
    }

    //detects test image features
    
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    this->detector->detectFeatures(src_img_filtered, keypoints, descriptors);
    SceneFeatures src_features(keypoints, descriptors);

    //matches test image features with every models' features and store them in out_matches
    std::vector<std::vector<cv::DMatch>> out_matches;
    for(ModelFeatures model_features : this->models_features){
        std::vector<cv::DMatch> out_matches_t;

        try{
            this->matcher->matchFeatures(model_features.descriptors, src_features.descriptors, out_matches_t);
        }
        catch (const CustomErrors::InvalidArgumentError& e) {
            std::cerr << "Warning: Error in matching features: " << e.what() << std::endl;
            continue; // Skip this model and continue with the next one
        }
        out_matches.push_back(out_matches_t);
    }

    //finds the best model (the one with the most matches)
    int best_model_idx = -1;
    size_t best_score = 0;
    for (size_t i = 0; i < out_matches.size(); ++i) {
        if (out_matches[i].size() > best_score) {
            best_score = out_matches[i].size();
            best_model_idx = static_cast<int>(i);
        }
    }

    if(best_model_idx == -1){
        std::cerr << "Warning: no suitable model found" << std::endl;
        return;
    }
    //calculates bounding box of the object found in the test image
    cv::Mat imgModel = Utils::Loader::load_image(this->dataset.get_models()[best_model_idx].first);
    cv::Mat maskModel = Utils::Loader::load_image(this->dataset.get_models()[best_model_idx].second);
    Label labelObj = findBoundingBox(out_matches[best_model_idx],  this->models_features[best_model_idx].keypoints, src_features.keypoints, imgModel,  maskModel, src_img_filtered, this->dataset.get_type());
    
    out_labels.push_back(labelObj);
}

Label FeaturePipeline::findBoundingBox(const std::vector<cv::DMatch>& matches,
    const std::vector<cv::KeyPoint>& model_keypoint,
    const std::vector<cv::KeyPoint>& scene_keypoint,
    const cv::Mat& img_model,
    const cv::Mat& mask_model,
    const cv::Mat& img_scene,
    Object_Type object_type) const 
{
    const int minMatches = 4;

    if (matches.size() < minMatches) {
        std::cout << "Warning: not enough matches are found - " << matches.size() << "/" << minMatches << std::endl;
        return Label(object_type, cv::Rect());
    }

    cv::Mat cropped_imgModel = img_model(cv::boundingRect(mask_model)); // crop the image to remove the white background of the mask
    cv::Mat cropped_maskModel = mask_model(cv::boundingRect(mask_model)); // crop the mask to remove the white background of the mask

    std::vector<cv::Point2f> scene_pts, model_pts;
    for (const auto& match : matches) {
        model_pts.push_back(model_keypoint[match.queryIdx].pt);
        scene_pts.push_back(scene_keypoint[match.trainIdx].pt);
    }
    
    
    cv::Mat homography_mask;
    cv::Mat H = cv::findHomography(model_pts, scene_pts, cv::RANSAC, 5.0, homography_mask);
    if (H.empty()){
        std::cerr << "Warning: homography matrix empty" << std::endl;
        return Label(object_type, cv::Rect());
    }
    

    cv::Rect mask_rect = cv::boundingRect(cropped_maskModel);
    std::vector<cv::Point2f> model_corners = {
        {static_cast<float>(mask_rect.x), static_cast<float>(mask_rect.y)},
        {static_cast<float>(mask_rect.x + mask_rect.width), static_cast<float>(mask_rect.y)},
        {static_cast<float>(mask_rect.x + mask_rect.width), static_cast<float>(mask_rect.y + mask_rect.height)},
        {static_cast<float>(mask_rect.x), static_cast<float>(mask_rect.y + mask_rect.height)}
    };
    
    std::vector<cv::Point2f> scene_corners;     //corners of the detected object in the scene (not a horizontal/vertical rectangle, but commonly rotated)
    cv::perspectiveTransform(model_corners, scene_corners, H);
    
    
    std::vector<cv::Point2i> scene_corners_int;
    for( int i = 0; i < scene_corners.size(); i++){
        scene_corners_int.push_back(cv::Point2i(scene_corners[i].x, scene_corners[i].y));      
    }
    cv::Rect sceneBB = cv::boundingRect(scene_corners);     //bounding box of the 4 scene corners obtained by the perspective transform (commonly way bigger than the former bounding box)
    
    /*
    cv::Mat img_scene_copy = img_scene.clone();
    cv::polylines(img_scene_copy, scene_corners_int, true, cv::Scalar(255, 0, 0), 5);   //BLUE draw the bounding box (rotated rectangle) on the image
    cv::rectangle(img_scene_copy, sceneBB, cv::Scalar(0, 255, 0), 5);                      //GREEN draw the bounding box (axis-aligned rectangle) on the image
    cv::imshow("test image /w bounding box", img_scene_copy);
    cv::Mat imgSceneMatches = img_scene.clone();
    cv::drawMatches( 
        cropped_imgModel,
        model_keypoint,
        img_scene,
        scene_keypoint,
        matches,
        imgSceneMatches,
        cv::Scalar(0, 255, 0),
        cv::Scalar(0, 0, 255),
        homography_mask
        );
    cv::imshow("matches", imgSceneMatches);
    cv::waitKey(0);
    */
   
    return Label(object_type, sceneBB);
}



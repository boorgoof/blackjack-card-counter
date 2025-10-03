#include "../../../../include/card_detector/objectDetector/featurePipeline/FeaturePipeline.h"

void FeaturePipeline::update_extractor_matcher_compatibility() {
    if (this->extractor->getType() == ExtractorType::Type::ORB && this->matcher->getType() == MatcherType::Type::FLANN) {
        this->matcher = std::make_unique<FeatureMatcher>(FeatureMatcher(MatcherType::Type::FLANN, new cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2))));
    }
}

FeaturePipeline::~FeaturePipeline() {}


FeaturePipeline::FeaturePipeline(FeatureExtractor* extractor, FeatureMatcher* matcher, const FeatureDescriptorAlgorithm algoDescriptor)
    : extractor{extractor}, matcher{matcher}, template_descriptors{ FeatureDescriptorAlgorithmUtils::get_templates_descriptors(algoDescriptor) }
{      
    this->update_extractor_matcher_compatibility();
    
    std::string method_name = ExtractorType::toString(extractor->getType()) + "-" + MatcherType::toString(matcher->getType());
    this->set_method_name(method_name);
}

void FeaturePipeline::detect_objects(const cv::Mat &src_img, const cv::Mat &src_mask, std::vector<Label> &out_labels) {

    out_labels.clear();

    //extracts test image features
    std::vector<cv::KeyPoint> src_img_keypoints;
    cv::Mat src_img_descriptors;
    this->extractor->extractFeatures(src_img, src_mask, src_img_keypoints, src_img_descriptors);

    //the template descriptors are already extracted and passed to the pipeline (they always remain the same for every test image, so they are detected only once)

    //matches test image features with every template's features and store them in out_matches
    std::map<Card_Type, std::vector<cv::DMatch>> out_matches;
    for(const auto& kv : this->template_descriptors){

        Card_Type card = kv.first;               
        const cv::Mat& template_descriptor = kv.second.getMat();;

        std::vector<cv::DMatch> card_matches;
        try{
            this->matcher->matchFeatures(template_descriptor, src_img_descriptors, card_matches);
        }
        catch(const cv::Exception& e){
            std::cerr << "Error during feature matching: " << e.what() << std::endl;
            continue;
        }

        out_matches[card] = std::move(card_matches);
    }

    Card_Type best_card{Card_Type("UNKNOWN")};
    size_t best_score = 0;
    bool found = false;

    for (const auto& [card, matches] : out_matches) {
        if (matches.size() > best_score) {
            best_score = matches.size();
            best_card  = card;
            found = true;
        }
    }

    if (!found) {
        std::cerr << "Warning: no suitable model found" << std::endl;
        return ;
    } 

    const auto& best_matches = out_matches.at(best_card);

    
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



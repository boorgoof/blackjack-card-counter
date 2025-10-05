#include "../../../../include/card_detector/objectDetector/featurePipeline/FeaturePipeline.h"
#include "../../../../include/card_detector/objectDetector/featurePipeline/features/FeatureContainer.h"
#include "../../../../include/card_detector/objectDetector/featurePipeline/features/KeypointFeature.h"
#include "../../../../include/Loaders.h"

void FeaturePipeline::update_extractor_matcher_compatibility() {
    if (this->extractor->getType() == ExtractorType::ORB && this->matcher->getType() == MatcherType::FLANN) {
        this->matcher = std::make_unique<FeatureMatcher>(FeatureMatcher(MatcherType::FLANN, new cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2))));
    }
}

FeaturePipeline::~FeaturePipeline() {}


FeaturePipeline::FeaturePipeline(FeatureExtractor* extractor, FeatureMatcher* matcher, const std::string& template_cards_folder_path)
    : extractor{extractor}, matcher{matcher}, template_features{Utils::FeatureContainer::get_templates_features(template_cards_folder_path, extractor)}
{
    this->update_extractor_matcher_compatibility();

    std::string method_name = ExtractorType::toString(extractor->getType()) + "-" + MatcherType::toString(matcher->getType());
    this->set_method_name(method_name);
}



void FeaturePipeline::detect_objects(const cv::Mat &src_img, const cv::Mat &src_mask, std::vector<Label> &out_labels) {

    out_labels.clear();

    //1) Extracts test image features
    std::vector<cv::KeyPoint> src_img_keypoints;
    cv::Mat src_img_descriptors;
    this->extractor->extractFeatures(src_img, src_mask, src_img_keypoints, src_img_descriptors);

    //2) The template descriptors are already extracted and passed to the pipeline in the constuctor(they always remain the same for every test image, so they are detected only once)

    //3) Matches test image features with every template's features and store them in out_matches
    std::map<Card_Type, std::vector<cv::DMatch>> out_matches;

    for (const auto& [card, feature] : this->template_features) {
        
        if (!feature) continue;

        const KeypointFeature* keypoint_features = dynamic_cast<const KeypointFeature*>(feature);
        if (!keypoint_features) {
            std::cerr << "The dynamic cast from Feature* to KeypointFeature is not possible for the card" << card << "\n";
            continue;
        }

        // get the descriptors of the template
        const cv::Mat& templ_desciptors = keypoint_features->getDescriptors();
        if (templ_desciptors.empty() || src_img_descriptors.empty()) continue;

        // obtains the matches between the template and the test image
        std::vector<cv::DMatch> matches_templ_img;
        try {

            this->matcher->matchFeatures(templ_desciptors, src_img_descriptors,  matches_templ_img);

        } catch (const cv::Exception& e) {
            std::cerr << "Error during feature matching: " << e.what() << '\n';
            continue;
        }

        out_matches.emplace(card, std::move( matches_templ_img));
    }

    // 4) find the best template (the one with the highest number of matches with the test image)
    Card_Type best_card{Card_Type("UNKNOWN")};
    size_t best_score = 0;
    bool template_found = false;

    for (const auto& [card, matches] : out_matches) {
        if (matches.size() > best_score) {
            best_score = matches.size();
            best_card  = card;
            template_found = true;
        }
    }
    if (!template_found) {
        std::cerr << "Warning: no good template found for the test image during the feature matching" << std::endl;
        return ;
    } 
    const std::vector<cv::DMatch>& best_matches = out_matches.at(best_card);

    //5) Calculates bounding box of the object found in the test image
    Label labelObj = findBoundingBox(best_matches,  , src_img_keypoints, ,  , src_img, best_card);
    
    out_labels.push_back(labelObj);
}

Label FeaturePipeline::findBoundingBox(const std::vector<cv::DMatch>& matches,
    const std::vector<cv::KeyPoint>& template_keypoint,
    const std::vector<cv::KeyPoint>& scene_keypoint,
    const cv::Mat& img_template,
    const cv::Mat& mask_template,
    const cv::Mat& img_scene,
    Card_Type card_template) const 
{
    const int minMatches = 4;

    if (matches.size() < minMatches) {
        std::cout << "Warning: not enough matches are found: " << matches.size() << ". Min: " << minMatches << std::endl;
        return Label(card_template, cv::Rect());
    }

    cv::Mat cropped_imgTemplate = img_template(cv::boundingRect(mask_template)); // crop the image to remove the white background of the mask
    cv::Mat cropped_maskTemplate = mask_template(cv::boundingRect(mask_template)); // crop the mask to remove the white background of the mask

    // 1) take the matched keypoints from the template and scene images
    std::vector<cv::Point2f> scene_pts, template_pts;
    for (const auto& match : matches) {
        template_pts.push_back(template_keypoint[match.queryIdx].pt);
        scene_pts.push_back(scene_keypoint[match.trainIdx].pt);
    }
    
    // 2) find the homography matrix between the template and the scene
    cv::Mat homography_mask;
    cv::Mat H = cv::findHomography(template_pts, scene_pts, cv::RANSAC, 5.0, homography_mask);
    if (H.empty()){
        std::cerr << "Warning: homography matrix empty" << std::endl;
        return Label(card_template, cv::Rect());
    }
    

    cv::Rect mask_rect = cv::boundingRect(cropped_maskTemplate);
    std::vector<cv::Point2f> template_corners = {
        {static_cast<float>(mask_rect.x), static_cast<float>(mask_rect.y)},
        {static_cast<float>(mask_rect.x + mask_rect.width), static_cast<float>(mask_rect.y)},
        {static_cast<float>(mask_rect.x + mask_rect.width), static_cast<float>(mask_rect.y + mask_rect.height)},
        {static_cast<float>(mask_rect.x), static_cast<float>(mask_rect.y + mask_rect.height)}
    };
    
    std::vector<cv::Point2f> scene_corners;     //corners of the detected object in the scene (not a horizontal/vertical rectangle, but commonly rotated)
    cv::perspectiveTransform(template_corners, scene_corners, H);
    
    
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
   
    return Label(card_template, sceneBB);
}



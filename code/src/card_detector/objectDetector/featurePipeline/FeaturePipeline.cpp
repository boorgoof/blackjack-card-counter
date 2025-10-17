#include "../../../../include/card_detector/objectDetector/featurePipeline/FeaturePipeline.h"
#include "../../../../include/card_detector/objectDetector/featurePipeline/features/FeatureContainer.h"
#include "../../../../include/card_detector/objectDetector/featurePipeline/features/KeypointFeature.h"
#include "../../../../include/StatisticsCalculation.h"
#include "../../../../include/Loaders.h"


void FeaturePipeline::update_extractor_matcher_compatibility() {
    /*
    if (this->extractor_->getType() == ExtractorType::ORB && this->matcher_->getType() == MatcherType::FLANN) {
        this->matcher_.release();
        this->matcher_ = std::make_unique<FeatureMatcher>(FeatureMatcher(MatcherType::FLANN, new cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2))));
    }*/
   if (this->extractor_->getType() == ExtractorType::ORB && this->matcher_->getType() == MatcherType::FLANN) {
        this->matcher_.release();
        this->matcher_ = std::make_unique<FeatureMatcher>(MatcherType::BRUTEFORCE_HAMMING,cv::BFMatcher::create(cv::NORM_HAMMING));
    }
    
}

FeaturePipeline::~FeaturePipeline() {}

FeaturePipeline::FeaturePipeline(FeatureExtractor *extractor, FeatureMatcher *matcher, const std::string &templates_folder_path, size_t minMatchesForRANSAC, size_t numMinInliers, double nmsIoU, double numRansacReprojErr, size_t numMaxInstancesPerTemplate)
    : extractor_{extractor}, matcher_{matcher}, template_features_{Utils::FeatureContainerSingleton::get_templates_features(templates_folder_path, *extractor)}
{
    minMatchesForRANSAC_ = minMatchesForRANSAC;
    numMinInliers_ = numMinInliers;
    nmsIoU_ = nmsIoU;
    numRansacReprojErr_ = numRansacReprojErr;
    numMaxInstancesPerTemplate_ = numMaxInstancesPerTemplate;

    this->update_extractor_matcher_compatibility();

    std::string method_name = ExtractorType::toString(extractor->getType()) + "-" + MatcherType::toString(matcher->getType());
    this->set_method_name(method_name);
}

FeaturePipeline::FeaturePipeline(const ExtractorType::FeatureDescriptorAlgorithm extractor, const MatcherType::MatcherAlgorithm matcher, const std::string &templates_folder_path, size_t minMatchesForRANSAC, size_t numMinInliers, double nmsIoU, double numRansacReprojErr, size_t numMaxInstancesPerTemplate)
    : extractor_{std::make_unique<FeatureExtractor>(extractor)}, matcher_{std::make_unique<FeatureMatcher>(matcher)}, template_features_{Utils::FeatureContainerSingleton::get_templates_features(templates_folder_path, *this->extractor_)}
{
    minMatchesForRANSAC_ = minMatchesForRANSAC;
    numMinInliers_ = numMinInliers;
    nmsIoU_ = nmsIoU;
    numRansacReprojErr_ = numRansacReprojErr;
    numMaxInstancesPerTemplate_ = numMaxInstancesPerTemplate;

    this->update_extractor_matcher_compatibility();

    std::string method_name = ExtractorType::toString(extractor) + "-" + MatcherType::toString(matcher);
    this->set_method_name(method_name);
}



/*
void FeaturePipeline::detect_objects(const cv::Mat &src_img, const cv::Mat &src_mask, std::vector<Label> &out_labels) {

    out_labels.clear();

    //1) Extracts test image features
    std::unique_ptr<KeypointFeature> imageFeatures(dynamic_cast<KeypointFeature*>(this->extractor_->extractFeatures(src_img, src_mask)));
    if (!imageFeatures) {
        std::cerr << "The dynamic cast from Feature* to KeypointFeature is not possible for the img ";
        return;
    }

    //2) The template descriptors are already extracted and passed to the pipeline in the constuctor(they always remain the same for every test image, so they are detected only once)

    //3) For each template, match its descriptors with the test image descriptors and find the bounding boxes of the templ_object in the test image
    std::map<ObjectType* , std::vector<cv::DMatch>> out_matches;
    for (const auto& [templ_object, templ_feature] : this->template_features_) {
        
        if (!templ_feature) continue;

        const KeypointFeature* templFeatures = dynamic_cast<const KeypointFeature*>(templ_feature);
        if (!templFeatures) {
            std::cerr << "The dynamic cast from Feature* to KeypointFeature is not possible for the object" << templ_object->get_id() << "\n";
            continue;
        }

        // get the descriptors of the template
        const cv::Mat& templ_desciptors = templFeatures->getDescriptors();
        if (templ_desciptors.empty() || imageFeatures->getDescriptors().empty()) continue;

        // obtains the matches between the template and the test image
        std::vector<cv::DMatch> matches;
        try {
            this->matcher_->matchFeatures(templ_desciptors, imageFeatures->getDescriptors(),  matches);
        } catch (const cv::Exception& e) {
            std::cerr << "Error during feature matching: " << e.what() << '\n';
            continue;
        }

        // find for the current template all the instances in the test image
        size_t intances_found= 0;
        while (matches.size() >= this->minMatchesForRANSAC_ && intances_found < this->numMaxInstancesPerTemplate_) {

             // note: good matches which provide correct estimation are called inliers and remaining are called outliers.
             // cv.findHomography() returns a mask which specifies the inlier and outlier points
             // https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html?utm_source=chatgpt.com
            std::vector<unsigned char> inlier_mask;
            Label label(templ_object->clone(), cv::Rect(), 0.f); 

            const bool bboxFound = this->findBoundingBox(
                matches,
                *templFeatures,
                *imageFeatures,
                label,
                inlier_mask
            );
            if (!bboxFound) break; // something went wrong during the bounding box finding
            
            // If the homograpy matrix is estimated correctly, we count the number of the inliers (match coherent with the homograpy) and if they
            // are less than the min threshold the estimate is considered too weak: we break the loop 
            size_t inliers = 0;
            for (size_t i = 0; i < inlier_mask.size(); ++i) if (inlier_mask[i]) ++inliers;
            if (inliers < this->numMinInliers_) break;

            // If we have enough inlier, we accept the label for one instance. 
            out_labels.push_back(std::move(label));
            ++intances_found;

            // then we removes the the inliers used so that another instance can be found in the next iteration.
            std::vector<cv::DMatch> remaining; remaining.reserve(matches.size());
            for (size_t i = 0; i < matches.size(); ++i) {
                if (!inlier_mask[i]) remaining.push_back(matches[i]);
            }
            matches.swap(remaining);
        }
 
    }

    if (out_labels.empty()) {
        std::cerr << " Warning: No instances found.\n";
        return;
    }

    //4) Apply Non-Maxima Suppression to the found bounding boxes to remove overlapping boxe
    nmsLabels(out_labels, this->nmsIoU_);
     
}*/


void FeaturePipeline::detect_objects(const cv::Mat &src_img, const cv::Mat &src_mask, std::vector<Label> &out_labels) {

    out_labels.clear();

    // --- TOGGLE VELOCE: metti a 0 per spegnere
    #define DBG_VIZ 1

    //1) Extracts test image features
    std::unique_ptr<KeypointFeature> imageFeatures(dynamic_cast<KeypointFeature*>(this->extractor_->extractFeatures(src_img, src_mask)));

    // === VIZ: input, mask, overlay, keypoints immagine ===
    #if DBG_VIZ
    if (!src_img.empty()) {
        cv::namedWindow("IMG input", cv::WINDOW_NORMAL);
        cv::imshow("IMG input", src_img);
    }
    if (!src_mask.empty()) {
        cv::namedWindow("MASK", cv::WINDOW_NORMAL);
        cv::imshow("MASK", src_mask);

        // overlay maschera (ciano) sull'immagine
        if (!src_img.empty()) {
            cv::Mat img3, maskU, colorMask, overlay;
            if (src_img.channels()==1) cv::cvtColor(src_img, img3, cv::COLOR_GRAY2BGR); else img3 = src_img.clone();
            src_mask.convertTo(maskU, CV_8U);
            colorMask = cv::Mat::zeros(img3.size(), img3.type());
            std::vector<cv::Mat> ch(3);
            ch[0] = cv::Mat::zeros(img3.size(), CV_8U);
            ch[1] = maskU; // G
            ch[2] = maskU; // R
            cv::merge(ch, colorMask);
            cv::addWeighted(img3, 1.0, colorMask, 0.35, 0.0, overlay);
            cv::namedWindow("IMG+MASK overlay", cv::WINDOW_NORMAL);
            cv::imshow("IMG+MASK overlay", overlay);
        }
    }

    if (imageFeatures) {
        cv::Mat img_kp_vis;
        const auto& img_kps = imageFeatures->getKeypoints();
        if (!src_img.empty()) {
            cv::drawKeypoints(src_img, img_kps, img_kp_vis, cv::Scalar::all(-1),
                              cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        } else {
            // se non hai l’immagine, crea una tela nera minimale
            float maxx=1, maxy=1; for (const auto& k: img_kps){ maxx=std::max(maxx,k.pt.x); maxy=std::max(maxy,k.pt.y); }
            cv::Mat canvas = cv::Mat::zeros((int)std::ceil(maxy)+20, (int)std::ceil(maxx)+20, CV_8UC3);
            cv::drawKeypoints(canvas, img_kps, img_kp_vis, cv::Scalar::all(-1),
                              cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        }
        cv::namedWindow("IMG keypoints", cv::WINDOW_NORMAL);
        cv::imshow("IMG keypoints", img_kp_vis);
    }

    cv::waitKey(0);

    #endif
    // === /VIZ ===

    //2) template descriptors già in constructor

    //3) per ogni template...
    std::map<ObjectType* , std::vector<cv::DMatch>> out_matches;
    for (const auto& [templ_object, templ_feature] : this->template_features_) {
        
        if (!templ_feature) continue;

        const KeypointFeature* templFeatures = dynamic_cast<const KeypointFeature*>(templ_feature);
        if (!templFeatures) {
            std::cerr << "The dynamic cast from Feature* to KeypointFeature is not possible for the object" << templ_object->get_id() << "\n";
            continue;
        }
        std::cout<< "number of template keypoint for" << templ_object->get_id() << " is "<<templFeatures->getKeypoints().size() << std::endl;

        
        // === VIZ: keypoints del template su una tela ===
        #if DBG_VIZ
        {
            const auto& tkps = templFeatures->getKeypoints();
            // canvas abbastanza grande per contenere i kp del template
            float maxx=1, maxy=1; for (const auto& k: tkps){ maxx=std::max(maxx,k.pt.x); maxy=std::max(maxy,k.pt.y); }
            cv::Mat templCanvas = cv::Mat::zeros((int)std::ceil(maxy)+20, (int)std::ceil(maxx)+20, CV_8UC3);
            cv::Mat templ_kp_vis;
            cv::drawKeypoints(templCanvas, tkps, templ_kp_vis, cv::Scalar::all(-1),
                              cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            std::string winName = "TEMPLATE keypoints id=" + templ_object->get_id();
            if(templ_object->get_id() == "10C"){
                cv::namedWindow(winName, cv::WINDOW_NORMAL);
                cv::imshow(winName, templ_kp_vis);
                cv::waitKey(0);
            }
           
        }
        #endif
        

        // === /VIZ ===

        // get the descriptors of the template
        const cv::Mat& templ_desciptors = templFeatures->getDescriptors();
        if (templ_desciptors.empty() || imageFeatures->getDescriptors().empty()) continue;

        // obtains the matches between the template and the test image
        std::vector<cv::DMatch> matches;
        try {
            this->matcher_->matchFeatures(templ_desciptors, imageFeatures->getDescriptors(),  matches);
        } catch (const cv::Exception& e) {
            std::cerr << "Error during feature matching: " << e.what() << '\n';
            continue;
        }

        
        #if DBG_VIZ
        {
            // riduci per visualizzazione
            std::vector<cv::DMatch> mm = matches;
            if (mm.size() > 500) mm.resize(500);

            // ridisegno canvas template
            const auto& tkps = templFeatures->getKeypoints();
            float maxx=1, maxy=1; for (const auto& k: tkps){ maxx=std::max(maxx,k.pt.x); maxy=std::max(maxy,k.pt.y); }
            cv::Mat templCanvas = cv::Mat::zeros((int)std::ceil(maxy)+20, (int)std::ceil(maxx)+20, CV_8UC3);

            cv::Mat vis_matches;
            cv::drawMatches(templCanvas, tkps,
                            src_img.empty()? templCanvas : src_img, imageFeatures->getKeypoints(),
                            mm, vis_matches,
                            cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                            cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            if(templ_object->get_id() == "10C"){
                cv::namedWindow("MATCHES (primi 500)", cv::WINDOW_NORMAL);
                cv::imshow("MATCHES (primi 500)", vis_matches);
                cv::waitKey(0);
            }
        }
        #endif
        std::cout<< templ_object->get_id() <<"numero di matches: " << matches.size() << std::endl;
        

        // find instances...
        size_t intances_found= 0;
        while (matches.size() >= this->minMatchesForRANSAC_ && intances_found < this->numMaxInstancesPerTemplate_) {

            std::vector<unsigned char> inlier_mask;
            Label label(templ_object->clone(), cv::Rect(), 0.f); 

            const bool bboxFound = this->findBoundingBox(
                matches,
                *templFeatures,
                *imageFeatures,
                label,
                inlier_mask
            );
            if (!bboxFound) break;

            size_t inliers = 0;
            for (size_t i = 0; i < inlier_mask.size(); ++i) if (inlier_mask[i]) ++inliers;
            if (inliers < this->numMinInliers_) break;

            out_labels.push_back(std::move(label));
            ++intances_found;

            // rimuovi inlier
            std::vector<cv::DMatch> remaining; remaining.reserve(matches.size());
            for (size_t i = 0; i < matches.size(); ++i) {
                if (!inlier_mask[i]) remaining.push_back(matches[i]);
            }
            matches.swap(remaining);
        }
    }

    if (out_labels.empty()) {
        std::cerr << " Warning: No instances found.\n";
        return;
    }

    //4) NMS
    nmsLabels(out_labels, this->nmsIoU_);

    // === VIZ: (opzionale) disegna le bbox finali sulla IMG ===
    #if DBG_VIZ
    if (!src_img.empty()) {
        cv::Mat final_vis;
        if (src_img.channels()==1) cv::cvtColor(src_img, final_vis, cv::COLOR_GRAY2BGR); else final_vis = src_img.clone();
        for (const auto& L : out_labels) {
            cv::rectangle(final_vis, L.get_bounding_box(), cv::Scalar(0,255,0), 2);
        }
        
        cv::namedWindow("DETECTIONS dopo NMS", cv::WINDOW_NORMAL);
        cv::imshow("DETECTIONS dopo NMS", final_vis);
        cv::waitKey(0);
    }
    #endif
}


bool FeaturePipeline::findBoundingBox(const std::vector<cv::DMatch>& matches,
                                      const KeypointFeature& templFeatures,       
                                      const KeypointFeature& imgFeatures,        
                                      Label& out_label,
                                      std::vector<unsigned char>& out_inlier_mask) const
{
    if (matches.size() < 4) return false; // need at least 4 matches to compute homography

    const std::vector<cv::KeyPoint>& templ_kp = templFeatures.getKeypoints();
    const std::vector<cv::KeyPoint>& image_kp = imgFeatures.getKeypoints();
    const std::vector<cv::Point2f>& templ_rect_corners = templFeatures.getRectPoints(); 

    if (templ_rect_corners.size() != 4) {
        std::cerr << "Rect points must be 4 (the corners of the template) for templ_object: " << out_label.get_object()->get_id() << "\n";
        return false;
    }

    // 1) take the matched keypoints from the template and scene images
    std::vector<cv::Point2f> templ_pts, image_pts;
    templ_pts.reserve(matches.size());
    image_pts.reserve(matches.size());
    for (const cv::DMatch& m : matches) {
        templ_pts.push_back(templ_kp[m.queryIdx].pt);
        image_pts.push_back(image_kp[m.trainIdx].pt);
    }

    //2) estimate the homography matrix between the template and the scene
    cv::Mat H = cv::findHomography(templ_pts, image_pts, cv::RANSAC, this->numRansacReprojErr_, out_inlier_mask);
    if (H.empty()) return false;
    if (cv::determinant(H) == 0) return false;

    //3) project the 4 corners of the template into the scene image. So we obtain the bounding box of the templ_object in the scene
    std::vector<cv::Point2f> image_corners;
    cv::perspectiveTransform(templ_rect_corners, image_corners, H);
    cv::Rect bbox = cv::boundingRect(image_corners);
    if (bbox.area() <= 0) return false;

    out_label.set_bounding_box(bbox);
    out_label.set_confidence(0.0f); // TODO
    return true;
}


void FeaturePipeline::nmsLabels(std::vector<Label>& labels, double iou_thresh) const {

    // Sort labels by area in descending order // TODO sarebbe meglio ordinare per score, non lo ho attualmente in caso gli inliers?
    std::sort(labels.begin(), labels.end(), [](const Label& a, const Label& b){
        return a.get_bounding_box().area() > b.get_bounding_box().area();
    });

    std::vector<Label> toKeep;
    for (Label& candidate : labels) {
        bool suppress = false;
        for (const auto& kept : toKeep) {
            if (StatisticsCalculation::calc_IoU(candidate, kept) >= iou_thresh) {
                suppress = true;
                break;
            }
        }
        if (!suppress) toKeep.push_back(std::move(candidate));
    }
    labels.swap(toKeep);
}




// todo delete
/*
FeaturePipeline::FeaturePipeline(FeatureExtractor* extractor, FeatureMatcher* matcher, const std::string& template_cards_folder_path)
    : extractor_{extractor}, matcher_{matcher}, template_features_{Utils::FeatureContainerSingleton::get_templates_features(template_cards_folder_path, *extractor)}
{
    this->update_extractor_matcher_compatibility();

    std::string method_name = ExtractorType::toString(extractor->getType()) + "-" + MatcherType::toString(matcher->getType());
    this->set_method_name(method_name);
}

FeaturePipeline::FeaturePipeline(const ExtractorType::FeatureDescriptorAlgorithm extractor, const MatcherType::MatcherAlgorithm matcher, const std::string& template_cards_folder_path)
    : extractor_{std::make_unique<FeatureExtractor>(extractor)}, matcher_{std::make_unique<FeatureMatcher>(matcher)}, template_features_{Utils::FeatureContainerSingleton::get_templates_features(template_cards_folder_path, *this->extractor_)}
{
    this->update_extractor_matcher_compatibility();

    std::string method_name = ExtractorType::toString(extractor) + "-" + MatcherType::toString(matcher);
    this->set_method_name(method_name);
}*/

#include "../../../../include/card_detector/objectDetector/featurePipeline/FeaturePipeline.h"
#include "../../../../include/card_detector/objectDetector/featurePipeline/features/FeatureContainer.h"
#include "../../../../include/card_detector/objectDetector/featurePipeline/features/KeypointFeature.h"
#include "../../../../include/StatisticsCalculation.h"
#include "../../../../include/Loaders.h"

void FeaturePipeline::update_extractor_matcher_compatibility() {
    if (this->extractor_->getType() == ExtractorType::ORB && this->matcher_->getType() == MatcherType::FLANN) {
        this->matcher_.release();
        this->matcher_ = std::make_unique<FeatureMatcher>(FeatureMatcher(MatcherType::FLANN, new cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2))));
    }
}

FeaturePipeline::~FeaturePipeline() {}


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
}



namespace {
    constexpr size_t minMatchesForRANSAC = 25; // matches needed to apply RANSAC 
    constexpr size_t numMinInliers = 15; // min inliers to validate the found bbox 
    constexpr double nmsIoU = 0.30; // non-maxima suppression IoU threshold
    constexpr double numRansacReprojErr = 3.0;
    constexpr size_t numMaxInstancesPerTemplate = 50;
}

void FeaturePipeline::detect_objects(const cv::Mat &src_img, const cv::Mat &src_mask, std::vector<Label> &out_labels) {

    out_labels.clear();

    //1) Extracts test image features
    std::unique_ptr<KeypointFeature> imageFeatures(dynamic_cast<KeypointFeature*>(this->extractor_->extractFeatures(src_img, src_mask)));

    //2) The template descriptors are already extracted and passed to the pipeline in the constuctor(they always remain the same for every test image, so they are detected only once)

    //3) For each template, match its descriptors with the test image descriptors and find the bounding boxes of the object in the test image
    std::map<Card_Type, std::vector<cv::DMatch>> out_matches;
    for (const auto& [card, feature] : this->template_features_) {
        
        if (!feature) continue;

        const KeypointFeature* templFeatures = dynamic_cast<const KeypointFeature*>(feature);
        if (!templFeatures) {
            std::cerr << "The dynamic cast from Feature* to KeypointFeature is not possible for the card" << card << "\n";
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
        while (matches.size() >= minMatchesForRANSAC && intances_found < numMaxInstancesPerTemplate) {

             // note: good matches which provide correct estimation are called inliers and remaining are called outliers.
             // cv.findHomography() returns a mask which specifies the inlier and outlier points
             // https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html?utm_source=chatgpt.com
            std::vector<unsigned char> inlier_mask;
            Label label;

            const bool bboxFound = this->findBoundingBox(
                matches,
                templFeatures,
                *imageFeatures,
                card,
                label,
                inlier_mask
            );
            if (!bboxFound) break; // something went wrong during the bounding box finding
            
            // If the homograpy matrix is estimated correctly, we count the number of the inliers (match coherent with the homograpy) and if they
            // are less than the min threshold the estimate is considered too weak: we break the loop 
            size_t inliers = 0;
            for (size_t i = 0; i < inlier_mask.size(); ++i) if (inlier_mask[i]) ++inliers;
            if (inliers < numMinInliers) break;

            // If we have enough inlier, we accept the label for one instance. 
            out_labels.push_back(label);
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
        std::cerr << "No instances found.\n";
        return;
    }

    //4) Apply Non-Maxima Suppression to the found bounding boxes to remove overlapping boxe
    nmsLabels(out_labels, nmsIoU);
     
}


bool FeaturePipeline::findBoundingBox(const std::vector<cv::DMatch>& matches,
                                      const KeypointFeature* templFeatures,       
                                      const KeypointFeature& imgFeatures,        
                                      const Card_Type& card_template,
                                      Label& out_label,
                                      std::vector<unsigned char>& out_inlier_mask) const
{
    if (!templFeatures) return false;
    if (matches.size() < 4) return false; // need at least 4 matches to compute homography

    const auto& templ_kp   = templFeatures->getKeypoints();
    const auto& image_kp   = imgFeatures.getKeypoints();
    const auto& templ_rect_corners   = templFeatures->getRectPoints(); 

    if (templ_rect_corners.size() != 4) {
        std::cerr << "Rect points must be 4 (the corners of the template) for card: " << card_template << "\n";
        return false;
    }

    // 1) take the matched keypoints from the template and scene images
    std::vector<cv::Point2f> templ_pts, image_pts;
    templ_pts.reserve(matches.size());
    image_pts.reserve(matches.size());
    for (const auto& m : matches) {
        templ_pts.push_back(templ_kp[m.queryIdx].pt);
        image_pts.push_back(image_kp[m.trainIdx].pt);
    }

    //2) estimate the homography matrix between the template and the scene
    cv::Mat H = cv::findHomography(templ_pts, image_pts, cv::RANSAC, numRansacReprojErr, out_inlier_mask);
    if (H.empty()) return false;

    //3) project the 4 corners of the template into the scene image. So we obtain the bounding box of the object in the scene
    std::vector<cv::Point2f> image_corners;
    cv::perspectiveTransform(templ_rect_corners, image_corners, H);
    cv::Rect bbox = cv::boundingRect(image_corners);
    if (bbox.area() <= 0) return false;

    out_label = Label(card_template, bbox);
    return true;
}


void FeaturePipeline::nmsLabels(std::vector<Label>& labels, double iou_thresh) const {

    // Sort labels by area in descending order // TODO sarebbe meglio ordinare per score, non lo ho attualmente in caso gli inliers?
    std::sort(labels.begin(), labels.end(), [](const Label& a, const Label& b){
        return a.get_bounding_box().area() > b.get_bounding_box().area();
    });

    std::vector<Label> toKeep;
    for (const auto& candidate : labels) {
        bool suppress = false;
        for (const auto& kept : toKeep) {
            if (StatisticsCalculation::calc_IoU(candidate, kept) >= iou_thresh) {
                suppress = true; break;
            }
        }
        if (!suppress) toKeep.push_back(candidate);
    }
    labels.swap(toKeep);
}

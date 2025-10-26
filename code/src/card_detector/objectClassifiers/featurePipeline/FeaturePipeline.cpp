#include "../../../../include/card_detector/objectClassifiers/featurePipeline/FeaturePipeline.h"
#include "../../../../include/Dataset/TemplateDataset.h"
#include "../../../../include/card_detector/objectClassifiers/featurePipeline/features/FeatureContainer.h"
#include "../../../../include/card_detector/objectClassifiers/featurePipeline/features/KeypointFeature.h"
#include "../../../../include/StatisticsCalculation.h"
#include "../../../../include/Loaders.h"



void FeaturePipeline::update_extractor_matcher_compatibility() {
    
    if (this->extractor_->getType() == ExtractorType::ORB && this->matcher_->getType() == MatcherType::FLANN) {
        this->matcher_.release();
        this->matcher_ = std::make_unique<FeatureMatcher>(FeatureMatcher(MatcherType::FLANN, new cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2))));
    }
}

FeaturePipeline::~FeaturePipeline() {}

FeaturePipeline::FeaturePipeline(FeatureExtractor *extractor, FeatureMatcher *matcher, TemplateDataset& template_dataset)
    : extractor_{extractor}, matcher_{matcher}, template_features_{Utils::FeatureContainerSingleton::get_templates_features(template_dataset, *extractor)}
{
    this->update_extractor_matcher_compatibility();

    std::string method_name = ExtractorType::toString(extractor->getType()) + "-" + MatcherType::toString(matcher->getType());
    this->set_method_name(method_name);
}

FeaturePipeline::FeaturePipeline(const ExtractorType::FeatureDescriptorAlgorithm extractor, const MatcherType::MatcherAlgorithm matcher, TemplateDataset& template_dataset)
    : extractor_{std::make_unique<FeatureExtractor>(extractor)}, matcher_{std::make_unique<FeatureMatcher>(matcher)}, template_features_{Utils::FeatureContainerSingleton::get_templates_features(template_dataset, *this->extractor_)}
{
    this->update_extractor_matcher_compatibility();

    std::string method_name = ExtractorType::toString(extractor) + "-" + MatcherType::toString(matcher);
    this->set_method_name(method_name);
}


const ObjectType* FeaturePipeline::classify_object(const cv::Mat &src_img, const cv::Mat &src_mask) {
    
    const ObjectType* best_obj = nullptr;

    //1) Extracts test image features
    std::unique_ptr<KeypointFeature> imageFeatures(dynamic_cast<KeypointFeature*>(this->extractor_->extractFeatures(src_img, src_mask)));
    if (!imageFeatures) { 
        std::cerr << "The dynamic cast from Feature* to KeypointFeature is not possible for the img ";
        return nullptr; 
    }

    //2) The template descriptors are already extracted and passed to the pipeline in the constuctor(they always remain the same for every test image, so they are detected only once)

    //3) For each template, match its descriptors with the test image descriptors and find the bounding boxes of the templ_object in the test image
    size_t best_score = 0;
    const size_t MIN_MATCHES_THRESHOLD = 10;
    for (const auto& [templ_object, templ_feature] : this->template_features_) {
        
        if (!templ_object || !templ_feature) continue;

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

        if (matches.size() >= MIN_MATCHES_THRESHOLD && matches.size() > best_score) {
            best_score = matches.size();
            best_obj = templ_object; 
        }
        
    }
    return best_obj;

}


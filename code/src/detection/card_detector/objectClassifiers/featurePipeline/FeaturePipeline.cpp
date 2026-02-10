#include "../../../../../include/detection/card_detector/objectClassifiers/featurePipeline/FeaturePipeline.h"
#include "../../../../../include/Dataset/TemplateDataset.h"
#include "../../../../../include/detection/card_detector/objectClassifiers/featurePipeline/features/FeatureContainer.h"

#include "../../../../../include/detection/card_detector/objectClassifiers/featurePipeline/extractors/KeypointExtractor.h"
#include "../../../../../include/detection/card_detector/objectClassifiers/featurePipeline/extractors/HashExtractor.h"
#include "../../../../../include/detection/card_detector/objectClassifiers/featurePipeline/matchers/KeypointMatcher.h"
#include "../../../../../include/detection/card_detector/objectClassifiers/featurePipeline/matchers/HashMatcher.h"


#include "../../../../../include/Utils.h"
#include "../../../../../include/StatisticsCalculation.h"
#include "../../../../../include/Loaders.h"

FeaturePipeline::~FeaturePipeline() {}

FeaturePipeline::FeaturePipeline(ExtractorType::FeatureDescriptorAlgorithm extractor_algorithm, const double lowe_ratio_threshold, TemplateDataset& template_dataset)
    : extractor_{nullptr}, matcher_{nullptr}, lowe_ratio_threshold_{lowe_ratio_threshold}, template_features_{nullptr}
{
    switch (extractor_algorithm) {
    case ExtractorType::SIFT:
        extractor_ = std::make_unique<KeypointExtractor>(extractor_algorithm);
        matcher_ = std::make_unique<KeypointMatcher>(MatcherType::FLANN, 10, 0.8f); // min 10 matches and lowe's ratio 0.8
        break;
    case ExtractorType::ORB:
        extractor_ = std::make_unique<KeypointExtractor>(extractor_algorithm);
        matcher_ = std::make_unique<KeypointMatcher>(MatcherType::BRUTEFORCE_HAMMING, 10, 0.8f); // min 10 matches and lowe's ratio 0.8
        break;
    case ExtractorType::PHASH:
        extractor_ = std::make_unique<HashExtractor>(extractor_algorithm);
        matcher_ = std::make_unique<HashMatcher>(MatcherType::PHASH, 15.0); // PHash distance threshold 15.0
        break;
    case ExtractorType::COLOR_MOMENT_HASH:
        extractor_ = std::make_unique<HashExtractor>(extractor_algorithm);
        matcher_ = std::make_unique<HashMatcher>(MatcherType::COLOR_MOMENT_HASH, 0.1); // Color Moment Hash distance threshold 0.1
        break;
    default:
        throw std::invalid_argument("Unsupported extractor type for FeaturePipeline");
    }

    // Populate the Singleton
    FeatureContainer::getInstance().loadTemplates(template_dataset, *extractor_);
    
    // Store reference to the map for classification
    template_features_ = &FeatureContainer::getInstance().getFeatures();

    std::string method_name = ExtractorType::toString(extractor_->getType()) + "-" + MatcherType::toString(matcher_->getType());
    this->set_method_name(method_name);
}


const ObjectType* FeaturePipeline::classify_object(const cv::Mat &src_img, const cv::Mat &src_mask) {

    //Extracts test image features
    std::unique_ptr<Feature> imageFeatures(this->extractor_->extractFeatures(src_img, src_mask));
    if (!imageFeatures) { 
        return nullptr;
    }

    //Utils::Visualization::showImage(src_img, "Classify Object - Input Image", 3000, 1.0);

    const ObjectType* best_obj = nullptr;
    double best_score = matcher_->getWorstScore();
    double second_best_score = matcher_->getWorstScore();

    //The template descriptors are already extracted and passed to the pipeline in the constuctor(they always remain the same for every test image, so they are detected only once)

    //For each template, match its descriptors with the test image descriptors and find the bounding boxes of the templ_object in the test image
    for (const auto& [templ_object, templ_feature] : *this->template_features_) {
        
        if (!templ_object || !templ_feature) continue;

        double current_score = matcher_->matchFeatures(imageFeatures.get(), templ_feature);

        if(matcher_->isValid(current_score)) {
            //std::cout << "Object " << templ_object->to_string() << " is a valid match with score: " << current_score << std::endl;

            if (matcher_->isBetter(current_score, best_score)) {
                second_best_score = best_score;
                best_score = current_score;
                best_obj = templ_object;
                //std::cout << "Object " << templ_object->to_string() << " is the best match so far with score: " << best_score << std::endl;
            }
            else if (matcher_->isBetter(current_score, second_best_score)) {
                second_best_score = current_score;
            }
        }
    }

    double ratio = matcher_->calculateRatio(best_score, second_best_score);
    //std::cout << "Best score: " << best_score << ", Second best score: " << second_best_score << ", Ratio: " << ratio << std::endl;

    if(ratio > this->lowe_ratio_threshold_) {
        std::cout << "No reliable match found. Best ratio " << ratio << " is above the threshold of " << this->lowe_ratio_threshold_ << std::endl;
        return nullptr;
    }

    return best_obj;
}

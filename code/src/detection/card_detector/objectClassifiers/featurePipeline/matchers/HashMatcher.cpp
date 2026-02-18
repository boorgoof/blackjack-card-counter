// Matteo Bino

#include "../../../../../../include/detection/card_detector/objectClassifiers/featurePipeline/matchers/HashMatcher.h"
#include "../../../../../../include/detection/card_detector/objectClassifiers/featurePipeline/features/HashFeature.h"

HashMatcher::HashMatcher(const MatcherType::MatcherAlgorithm &type, double threshold)
    : FeatureMatcher(type), threshold_{threshold} {
    switch (type) {
        case MatcherType::PHASH:
            hasher_ = cv::img_hash::PHash::create();
            break;
        case MatcherType::COLOR_MOMENT_HASH:
            hasher_ = cv::img_hash::ColorMomentHash::create();
            break;
        default:
            throw std::invalid_argument("Unsupported matcher type for HashMatcher");
    }
}

double HashMatcher::matchFeatures(const Feature *query, const Feature *target) const
{
    auto q = dynamic_cast<const HashFeature*>(query);
    auto t = dynamic_cast<const HashFeature*>(target);
    if (!q || !t) return std::numeric_limits<double>::max();

    return hasher_->compare(q->getHash(), t->getHash());
}

double HashMatcher::calculateRatio(double best, double second_best) const
{
    //scores go from 0 (best) to +inf (worst)
    if (second_best == 0.0) {
        return 1; // Avoid division by zero
    }
    return best / second_best;
}

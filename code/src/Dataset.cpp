#include "../Dataset.h"

#include <array>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <opencv2/imgcodecs.hpp>

#include "Loaders.h"

namespace {
constexpr const char* kImagesRelativePath = "../../Dataset/Images/Images/";
constexpr const char* kAnnotationsRelativePath = "../../Dataset/YOLO_Annotations/YOLO_Annotations/";
constexpr std::array<const char*, 13> kRanks = {"A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"};
constexpr std::array<char, 4> kSuits = {'C', 'D', 'H', 'S'};
constexpr int kCopiesPerCard = 51;
}

Dataset::Dataset()
    : Dataset(std::filesystem::path(kImagesRelativePath), std::filesystem::path(kAnnotationsRelativePath)) {}

Dataset::Dataset(std::filesystem::path image_root, std::filesystem::path annotation_root)
    : image_root_{std::move(image_root)},
      annotation_root_{std::move(annotation_root)},
      entries_{build_entries(image_root_, annotation_root_)} {}

bool Dataset::has_next() const noexcept {
    return current_index_ < entries_.size();
}

ImageInfo Dataset::next() {
    if (!has_next()) {
        throw std::out_of_range("Dataset::next called past the end of the dataset");
    }

    const Entry& entry = entries_[current_index_++];

    cv::Mat image = cv::imread(entry.image_path.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + entry.image_path.string());
    }

    std::vector<Label> labels;
    if (std::filesystem::exists(entry.annotation_path)) {
        labels = AnnotationLoaders::load_yolo_image_annotations(entry.annotation_path.string(), image.cols, image.rows);
    }

    return ImageInfo{std::move(image), entry.image_path.string(), std::move(labels)};
}

void Dataset::reset() noexcept {
    current_index_ = 0;
}

std::vector<Dataset::Entry> Dataset::build_entries(const std::filesystem::path& image_root,
                                                   const std::filesystem::path& annotation_root) {
    std::vector<Entry> entries;
    entries.reserve(kRanks.size() * kSuits.size() * kCopiesPerCard);

    for (const auto* rank : kRanks) {
        for (char suit : kSuits) {
            for (int copy = 0; copy < kCopiesPerCard; ++copy) {
                std::ostringstream stem;
                stem << rank << suit << copy;

                const auto image_path = image_root / (stem.str() + ".jpg");
                const auto annotation_path = annotation_root / (stem.str() + ".txt");

                if (!std::filesystem::exists(image_path)) {
                    continue;
                }

                entries.push_back({image_path, annotation_path});
            }
        }
    }

    return entries;
}

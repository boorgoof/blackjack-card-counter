#include "../include/Dataset.h"

#include <sstream>
#include <stdexcept>
#include <utility>
#include <filesystem>
#include <algorithm>
#include <cctype>

namespace {
constexpr const char* kImagesRelativePath = "Dataset/Images/Images/";
constexpr const char* kAnnotationsRelativePath = "Dataset/YOLO_Annotations/YOLO_Annotations/";
}

Dataset::Dataset()
    : Dataset(std::filesystem::path(kImagesRelativePath), std::filesystem::path(kAnnotationsRelativePath)) {}

Dataset::Dataset(const std::string& image_dir, const std::string& annotation_dir)
    : Dataset(std::filesystem::path(image_dir), std::filesystem::path(annotation_dir)) {}

Dataset::Dataset(std::filesystem::path image_root, std::filesystem::path annotation_root)
    : image_root_{std::move(image_root)},
      annotation_root_{std::move(annotation_root)},
      entries_{build_entries(image_root_, annotation_root_)} {}


std::vector<ImageInfo> Dataset::build_entries(const std::filesystem::path& image_root, const std::filesystem::path& annotation_root) {
    std::vector<ImageInfo> entries;
    if (!std::filesystem::exists(image_root) || !std::filesystem::is_directory(image_root)) {
        return entries;
    }

    // Collect all image files in the directory
    for (const auto& dirent : std::filesystem::directory_iterator(image_root)) {
        if (!dirent.is_regular_file()) continue;

        const auto& p = dirent.path();
        auto ext = p.extension().string();

        if (ext == ".jpg" || ext == ".png") { //jpg dataset given, png for dataset of the video
            const auto stem = p.stem().string();
            const auto ann = annotation_root / (stem + ".txt");
            entries.emplace_back(stem, p.string(), ann.string());
        }
    }

    // Sort deterministically by filename
    std::sort(entries.begin(), entries.end(), [](const ImageInfo& a, const ImageInfo& b){
        return a.name() < b.name();
    });

    return entries;
}

ImageInfo Dataset::at(std::size_t index) const {
    if (index >= entries_.size()) {
        throw std::out_of_range("Dataset::at index out of range");
    }
    return entries_.at(index);
}


Dataset& Dataset::operator++() {
    if (current_index_ < entries_.size()) {
        ++current_index_;
    }
    return *this;
}

Dataset Dataset::operator++(int) {
    Dataset tmp = *this;
    ++(*this);
    return tmp;
}

bool Dataset::operator==(const Dataset& other) const noexcept {
    return current_index_ == other.current_index_
        && image_root_ == other.image_root_
        && annotation_root_ == other.annotation_root_;
}

#include "../include/Dataset.h"
#include "../include/Utils.h"

#include <sstream>
#include <stdexcept>
#include <utility>
#include <filesystem>
#include <algorithm>
#include <cctype>

Dataset::Dataset(const std::string& dataset_path, const bool is_sequential = false)
    : Dataset(std::filesystem::path(dataset_path) / "Images" / "Images",
              std::filesystem::path(dataset_path) / "YOLO_Annotations" / "YOLO_Annotations",
              is_sequential) {
    this->path_ = std::filesystem::path(dataset_path);
}

Dataset::Dataset(const std::string& image_dir, const std::string& annotation_dir, const bool is_sequential = false)
    : Dataset(std::filesystem::path(image_dir), std::filesystem::path(annotation_dir), is_sequential) {
    //find common path between image_dir and annotation_dir
    this->path_ = Utils::Path::longestCommonPath(image_dir, annotation_dir);
}

Dataset::Dataset(std::filesystem::path image_root, std::filesystem::path annotation_root, const bool is_sequential = false)
    : image_root_{std::move(image_root)},
      annotation_root_{std::move(annotation_root)},
      entries_{build_entries(image_root_, annotation_root_)},
      is_sequential_{is_sequential} {}


std::vector<ImageInfo> Dataset::build_entries(const std::filesystem::path& image_root, const std::filesystem::path& annotation_root) {
    std::vector<ImageInfo> entries;
    if (!std::filesystem::exists(image_root) || !std::filesystem::is_directory(image_root)) {
        return entries;
    }

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

    std::sort(entries.begin(), entries.end(), [](const ImageInfo& a, const ImageInfo& b){
        return a.get_name() < b.get_name();
    });

    return entries;
}

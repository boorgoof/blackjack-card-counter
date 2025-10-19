#include "../../include/Dataset/ImageDataset.h"

#include <utility>
#include <filesystem>
#include <algorithm>

ImageDataset::ImageDataset(const std::string& dataset_path = "../data/datasets")
    : ImageDataset(std::filesystem::path(dataset_path) / "Images" / "Images",
                   std::filesystem::path(dataset_path) / "YOLO_Annotations" / "YOLO_Annotations") { }

ImageDataset::ImageDataset(const std::string& image_dir, const std::string& annotation_dir)
    : ImageDataset(std::filesystem::path(image_dir), std::filesystem::path(annotation_dir)) { }

ImageDataset::ImageDataset(std::filesystem::path image_root, std::filesystem::path annotation_root)
    : image_root_{image_root},
      annotation_root_{annotation_root},
      entries_{build_entries(image_root, annotation_root)} { }

Dataset::Iterator ImageDataset::begin() const {
    if (entries_.empty()) {
        return Iterator(nullptr);
    }
    ImageInfo* first = const_cast<ImageInfo*>(&entries_.front());
    return Iterator(first);
}

Dataset::Iterator ImageDataset::end() const {
    if (entries_.empty()) {
        return Iterator(nullptr);
    }
    ImageInfo* last = const_cast<ImageInfo*>(&entries_.back());
    ImageInfo* one_past_last = last + 1;
    return Iterator(one_past_last);
}

std::vector<ImageInfo> ImageDataset::build_entries(const std::filesystem::path& image_root, const std::filesystem::path& annotation_root) {
    std::vector<ImageInfo> entries;
    
    if (!std::filesystem::exists(image_root) || !std::filesystem::is_directory(image_root)) {
        return entries;
    }
    if (!std::filesystem::exists(annotation_root) || !std::filesystem::is_directory(annotation_root)) {
        return entries;
    }

    entries.reserve(std::distance(std::filesystem::directory_iterator(image_root), std::filesystem::directory_iterator{}));

    for (const std::filesystem::directory_entry& dirent : std::filesystem::directory_iterator(image_root)) {
        if (!dirent.is_regular_file()) continue;

        const std::filesystem::path& p = dirent.path();
        std::string ext = p.extension().string();
        
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
            return std::tolower(c);
        });

        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
            const std::string stem = p.stem().string();
            const std::filesystem::path ann = annotation_root / (stem + ".txt");
            entries.emplace_back(stem, p.string(), ann.string());
        }
    }

    std::sort(entries.begin(), entries.end(), [](const ImageInfo& a, const ImageInfo& b){
        return a.get_name() < b.get_name();
    });

    return entries;
}

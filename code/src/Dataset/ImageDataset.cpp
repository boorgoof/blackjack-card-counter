#include "../../include/Dataset/ImageDataset.h"
#include "../../include/SampleInfo/ImageInfo.h"
#include "../../include/Loaders.h"

#include <utility>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <memory>

ImageDataset::ImageDataset(const std::string& dataset_path)
    : ImageDataset(std::filesystem::path(dataset_path) / "Images" / "Images", std::filesystem::path(dataset_path) / "YOLO_Annotations" / "YOLO_Annotations") { }

ImageDataset::ImageDataset(const std::string& image_dir, const std::string& annotation_dir)
    : ImageDataset(std::filesystem::path(image_dir), std::filesystem::path(annotation_dir)) { }

ImageDataset::ImageDataset(std::filesystem::path image_root, std::filesystem::path annotation_root)
    : image_root_{image_root},
      annotation_root_{annotation_root}
{ 
    entries_ = build_entries();
}

Dataset::Iterator ImageDataset::begin() const {
    return Iterator(entries_.cbegin());
}

Dataset::Iterator ImageDataset::end() const {
    return Iterator(entries_.cend());
}

cv::Mat ImageDataset::load(const Dataset::Iterator& it) {
    if (entries_.empty() || it == Iterator(entries_.cend())) {
        std::cerr << "ImageDataset: invalid iterator or empty dataset" << std::endl;
        return {};
    }
    const SampleInfo& sample = *it;
    cv::Mat image = Loader::Image::load_image(sample.get_pathSample());
    if (image.empty()) {
        std::cerr << "ImageDataset: failed to load image from " << sample.get_pathSample() << std::endl;
    }
    return image;
}

std::vector<std::shared_ptr<SampleInfo>> ImageDataset::build_entries() {
    std::vector<std::shared_ptr<SampleInfo>> entries;

    if (!std::filesystem::exists(image_root_) || !std::filesystem::is_directory(image_root_)) {
        std::cerr << "ImageDataset: image root directory does not exist or is not a directory: " << image_root_ << std::endl;
        return entries;
    }
    if (!std::filesystem::exists(annotation_root_) || !std::filesystem::is_directory(annotation_root_)) {
        std::cerr << "ImageDataset: annotation root directory does not exist or is not a directory: " << annotation_root_ << std::endl;
        return entries;
    }

    entries.reserve(std::distance(std::filesystem::directory_iterator(image_root_), std::filesystem::directory_iterator{}));

    for (const std::filesystem::directory_entry& dirent : std::filesystem::directory_iterator(image_root_)) {
        if (!dirent.is_regular_file()) continue;

        const std::filesystem::path& p = dirent.path();
        std::string ext = p.extension().string();
        
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
            return std::tolower(c);
        });

        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
            const std::string stem = p.stem().string();
            const std::filesystem::path ann = annotation_root_ / (stem + ".txt");
            entries.emplace_back(std::make_unique<ImageInfo>(stem, p.string(), ann.string()));
        }
    }

    std::sort(entries.begin(), entries.end(), [](const std::shared_ptr<SampleInfo>& a, const std::shared_ptr<SampleInfo>& b){
        return a->get_name() < b->get_name();
    });

    return entries;
}

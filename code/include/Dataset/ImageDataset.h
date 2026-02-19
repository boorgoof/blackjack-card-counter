// Gianluca Caregnato

#ifndef IMAGEDATASET_H
#define IMAGEDATASET_H

#include "Dataset.h"
#include <filesystem>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

/**
 * @brief Dataset implementation for folder-based image collections.
 *
 * Manages datasets stored as image files in a directory,
 * with corresponding annotation files in a separate directory.
 */
class ImageDataset : public Dataset {
public:
    /**
     * @brief Construct from a single dataset path (legacy layout).
     * @param dataset_path Base path containing "Images/Images" and "YOLO_Annotations/YOLO_Annotations" subdirectories.
     */
    ImageDataset(const std::string& dataset_path);

    /**
     * @brief Construct from separate image and annotation directory strings.
     * @param image_dir Path to the directory containing images.
     * @param annotation_dir Path to the directory containing annotations.
     */
    ImageDataset(const std::string& image_dir, const std::string& annotation_dir);

    /**
     * @brief Construct from filesystem paths.
     * @param image_root Path to the directory containing images.
     * @param annotation_root Path to the directory containing annotations.
     */
    ImageDataset(std::filesystem::path image_root, std::filesystem::path annotation_root);

    ~ImageDataset() override = default;

    Iterator begin() const override;
    Iterator end() const override;
    size_t size() const noexcept override { return entries_.size(); }
    bool is_sequential() const noexcept override { return false; }
    std::filesystem::path get_root() const override { return image_root_; }
    std::filesystem::path get_annotation_root() const override { return annotation_root_; }
    cv::Mat load(const Iterator& it) override;

private:
    /**
     * @brief Scan image_root_ and annotation_root_ to build the sample entries.
     * @return Vector of SampleInfo shared pointers, sorted by name.
     */
    std::vector<std::shared_ptr<SampleInfo>> build_entries();

    std::vector<std::shared_ptr<SampleInfo>> entries_;
    std::filesystem::path image_root_;
    std::filesystem::path annotation_root_;
};

#endif // IMAGEDATASET_H

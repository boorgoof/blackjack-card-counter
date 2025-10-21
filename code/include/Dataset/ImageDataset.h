#ifndef IMAGEDATASET_H
#define IMAGEDATASET_H

#include "Dataset.h"
#include <filesystem>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

/**
 * @brief Concrete implementation of Dataset for folder-based image datasets.
 * 
 * This class manages datasets stored as collections of image files in a directory,
 * with corresponding annotation files in a separate directory.
 */
class ImageDataset : public Dataset {
public:
    /**
     * @brief Construct from dataset path (legacy).
     * @param dataset_path Base path containing "Images/Images" and "YOLO_Annotations/YOLO_Annotations" subdirectories.
     */
    ImageDataset(const std::string& dataset_path);
    
    /**
     * @brief Construct from separate image and annotation directories.
     * @param image_dir Path to the directory containing images.
     * @param annotation_dir Path to the directory containing annotations.
     * @param is_sequential Whether this is a sequential dataset (default: false).
     */
    ImageDataset(const std::string& image_dir, const std::string& annotation_dir);
    
    /**
     * @brief Construct from filesystem paths.
     * @param image_root Path to the directory containing images.
     * @param annotation_root Path to the directory containing annotations.
     * @param is_sequential Whether this is a sequential dataset (default: false).
     */
    ImageDataset(std::filesystem::path image_root, std::filesystem::path annotation_root);

    /**
     * @brief Default destructor.
     */
    ~ImageDataset() override = default;

    // Implement pure virtual methods from Dataset
    Iterator begin() const override;
    Iterator end() const override;
    size_t size() const noexcept override { return entries_.size(); }
    bool is_sequential() const noexcept override { return false; }
    std::filesystem::path get_root() const override { return image_root_; }
    std::filesystem::path get_annotation_root() const override { return annotation_root_; }
    cv::Mat load(const Iterator& it) override;

private:
    /**
     * @brief Builds the dataset entries by scanning the image and annotation directories.
     * @param image_root Path to the directory containing images.
     * @param annotation_root Path to the directory containing annotations.
     * @return A vector of SampleInfo objects representing the dataset entries.
     */
    static std::vector<std::shared_ptr<SampleInfo>> build_entries(const std::filesystem::path& image_root, const std::filesystem::path& annotation_root);

    std::vector<std::shared_ptr<SampleInfo>> entries_; // Vector of all sample info entries
    std::filesystem::path image_root_; // Root directory for images
    std::filesystem::path annotation_root_; // Root directory for annotations
};

#endif // IMAGEDATASET_H

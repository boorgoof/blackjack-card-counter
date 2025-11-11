#ifndef TEMPLATE_DATASET_H
#define TEMPLATE_DATASET_H 

#include "Dataset.h"
#include "../../include/Loaders.h"
#include "../CardType.h"
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

class TemplateDataset : public Dataset {
public:
    TemplateDataset(const std::string& dataset_path);
    ~TemplateDataset() override = default;

    Iterator begin() const override;
    Iterator end() const override;
    size_t size() const noexcept override { return entries_.size(); }
    bool is_sequential() const noexcept override { return false; }
    std::filesystem::path get_root() const override { return image_root_; }
    std::filesystem::path get_annotation_root() const override { static std::string s{}; return s; }
    cv::Mat load(const Iterator& it) override;
private:
    std::vector<std::shared_ptr<SampleInfo>> build_entries();
    std::vector<std::shared_ptr<SampleInfo>> entries_; // Vector of all sample info entries
    std::filesystem::path image_root_; // Root directory for images
};

#endif // TEMPLATE_DATASET_H

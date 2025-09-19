#ifndef DATASET_H
#define DATASET_H

#include <cstddef>
#include <filesystem>
#include <vector>
#include <string>

#include "ImageInfo.h"

class Dataset {
public:
    Dataset();
    // Construct from directories (preferred)
    Dataset(const std::string& image_dir, const std::string& annotation_dir);
    // Legacy/path overload
    Dataset(std::filesystem::path image_root, std::filesystem::path annotation_root);

    bool has_next() const noexcept { return current_index_ < entries_.size(); }
    ImageInfo next() { return at(current_index_++); }

    void reset() noexcept { current_index_ = 0; }

    std::size_t size() const noexcept { return entries_.size(); }
    std::size_t current_index() const noexcept { return current_index_; }

    const std::filesystem::path& image_root() const noexcept { return image_root_; }
    const std::filesystem::path& annotation_root() const noexcept { return annotation_root_; }

    // Indexed accessors
    ImageInfo at(std::size_t index) const;
    ImageInfo operator[](std::size_t index) const { return at(index); }

    // Iterator-like increment operators (advance dataset cursor)
    Dataset& operator++();
    Dataset operator++(int);

    // Position comparison (same roots and same cursor)
    bool operator==(const Dataset& other) const noexcept;
    bool operator!=(const Dataset& other) const noexcept { return !(*this == other); }

private:
    static std::vector<ImageInfo> build_entries(const std::filesystem::path& image_root,
                                                const std::filesystem::path& annotation_root);


    std::filesystem::path image_root_;
    std::filesystem::path annotation_root_;
    std::vector<ImageInfo> entries_;
    std::size_t current_index_{0};
};

#endif 

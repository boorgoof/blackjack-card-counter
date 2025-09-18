#ifndef DATASET_H
#define DATASET_H

#include <cstddef>
#include <filesystem>
#include <vector>

#include "ImageInfo.h"

class Dataset {
public:
    Dataset();
    Dataset(std::filesystem::path image_root, std::filesystem::path annotation_root);

    bool has_next() const noexcept;
    ImageInfo next();

    void reset() noexcept;

    std::size_t size() const noexcept { return entries_.size(); }
    std::size_t current_index() const noexcept { return current_index_; }

    const std::filesystem::path& image_root() const noexcept { return image_root_; }
    const std::filesystem::path& annotation_root() const noexcept { return annotation_root_; }

    // Indexed accessors
    ImageInfo at(std::size_t index) const;           // bounds-checked
    ImageInfo operator[](std::size_t index) const;    // unchecked

    // Iterator-like increment operators (advance dataset cursor)
    Dataset& operator++();       // prefix
    Dataset operator++(int);     // postfix

    // Position comparison (same roots and same cursor)
    bool operator==(const Dataset& other) const noexcept;
    bool operator!=(const Dataset& other) const noexcept { return !(*this == other); }

private:
    struct Entry {
        std::filesystem::path image_path;
        std::filesystem::path annotation_path;
    };

    static std::vector<Entry> build_entries(const std::filesystem::path& image_root,
                                            const std::filesystem::path& annotation_root);

    ImageInfo load_index(std::size_t index) const; // helper to materialize an item

    std::filesystem::path image_root_;
    std::filesystem::path annotation_root_;
    std::vector<Entry> entries_;
    std::size_t current_index_{0};
    bool is_sequential_;
};

#endif 

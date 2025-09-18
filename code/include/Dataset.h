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

private:
    struct Entry {
        std::filesystem::path image_path;
        std::filesystem::path annotation_path;
    };

    static std::vector<Entry> build_entries(const std::filesystem::path& image_root,
                                            const std::filesystem::path& annotation_root);

    std::filesystem::path image_root_;
    std::filesystem::path annotation_root_;
    std::vector<Entry> entries_;
    std::size_t current_index_{0};
};

#endif // DATASET_H

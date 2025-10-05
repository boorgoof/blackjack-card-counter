#ifndef DATASET_H
#define DATASET_H

#include <cstddef>
#include <filesystem>
#include <vector>
#include <string>

#include "ImageInfo.h"

class Dataset {
public:
    struct Iterator {
        using iterator_category = std::forward_iterator_tag;
        using difference_type   = std::ptrdiff_t;
        using value_type        = ImageInfo;
        using pointer           = ImageInfo*;  // or also value_type*
        using reference         = ImageInfo&;  // or also value_type&

    public:
        Iterator(pointer m_ptr) : m_ptr(m_ptr) {}

        reference operator*() const { return *m_ptr; }
        pointer operator->() { return m_ptr; }

        Iterator& operator++() { m_ptr++; return *this; }  

        Iterator operator++(int) { Iterator tmp = *this; ++(*this); return tmp; }

        friend bool operator== (const Iterator& a, const Iterator& b) { return a.m_ptr == b.m_ptr; };
        friend bool operator!= (const Iterator& a, const Iterator& b) { return a.m_ptr != b.m_ptr; };
    private:
        pointer m_ptr;
    };

    Dataset() = delete;

    Dataset(const std::string& dataset_path, const bool is_sequential);
    // Construct from directories (preferred)
    Dataset(const std::string& image_dir, const std::string& annotation_dir, const bool is_sequential);
    // Legacy/path overload
    Dataset(std::filesystem::path image_root, std::filesystem::path annotation_root, const bool is_sequential);

    Iterator begin() { return  Iterator(&entries_.front()); }
    Iterator end() { return Iterator(&entries_.back() + 1); }

    const bool is_sequential() const noexcept { return is_sequential_; }
    void set_is_sequential(bool val) noexcept { is_sequential_ = val; }
    const std::filesystem::path& get_path() const noexcept { return path_; }
    const std::filesystem::path& get_image_root() const noexcept { return image_root_; }
    const std::filesystem::path& get_annotation_root() const noexcept { return annotation_root_; }

private:
    static std::vector<ImageInfo> build_entries(const std::filesystem::path& image_root,
                                                const std::filesystem::path& annotation_root);

    std::filesystem::path path_;
    std::filesystem::path image_root_;
    std::filesystem::path annotation_root_;
    std::vector<ImageInfo> entries_;
    bool is_sequential_;
};




#endif 

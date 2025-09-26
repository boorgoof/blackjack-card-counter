#ifndef DATASET_H
#define DATASET_H

#include <cstddef>
#include <filesystem>
#include <vector>
#include <string>

#include "ImageInfo.h"

class Dataset {
public:
    /**
     * A forward iterator for Dataset entries.
     * This iterator allows traversal over the ImageInfo entries in the Dataset.
     * It supports standard iterator operations such as dereferencing, incrementing,
     * and comparison.
     */
    struct Iterator {
        using iterator_category = std::forward_iterator_tag;
        using difference_type   = std::ptrdiff_t;
        using value_type        = ImageInfo;
        using pointer           = ImageInfo*; 
        using reference         = ImageInfo&; 

    public:
        /**
         * Constructor for the Iterator.
         * @param m_ptr Pointer to the current ImageInfo entry.
         */
        Iterator(pointer m_ptr) : m_ptr(m_ptr) {}

        /**
         * @brief Dereference operator to access the current ImageInfo.
         * @return Reference to the current ImageInfo.
         */
        reference operator*() const { return *m_ptr; }

        /**
         * @brief Arrow operator to access members of the current ImageInfo.
         * @return Pointer to the current ImageInfo.
         */
        pointer operator->() { return m_ptr; }

        /**
         * @brief Pre-increment operator to move to the next ImageInfo.
         * @return Reference to the incremented iterator.
         */
        Iterator& operator++() { m_ptr++; return *this; }  

        /**
         * @brief Post-increment operator to move to the next ImageInfo.
         * @return Iterator before incrementing.
         */
        Iterator operator++(int) { Iterator tmp = *this; ++(*this); return tmp; }

        friend bool operator== (const Iterator& a, const Iterator& b) { return a.m_ptr == b.m_ptr; };
        friend bool operator!= (const Iterator& a, const Iterator& b) { return a.m_ptr != b.m_ptr; };
    private:
        pointer m_ptr;
    };

    Dataset() = delete;
    // Construct from dataset path (legacy)
    Dataset(const std::string& dataset_path, const bool is_sequential = false);
    // Construct from directories (preferred)
    Dataset(const std::string& image_dir, const std::string& annotation_dir, const bool is_sequential = false);
    // Legacy/path overload
    Dataset(std::filesystem::path image_root, std::filesystem::path annotation_root, const bool is_sequential = false);
    
    /**
     * @brief Returns an iterator to the beginning of the Dataset entries.
     * @return Iterator to the first ImageInfo entry.
     */
    Iterator begin() const { return entries_.empty() ? Iterator(nullptr) : Iterator(const_cast<ImageInfo*>(&entries_.front())); }

    /**
     * @brief Returns an iterator to the end of the Dataset entries.
     * @return Iterator to one past the last ImageInfo entry.
     */
    Iterator end() const { return entries_.empty() ? Iterator(nullptr) : Iterator(const_cast<ImageInfo*>(&entries_.back()) + 1); }
    

    /**
     * @brief Returns the number of entries in the Dataset.
     * @return Size of the Dataset.
     */
    size_t size() const noexcept { return entries_.size(); }

    /**
     * @brief Checks if the Dataset is empty.
     * @return True if the Dataset has no entries, false otherwise.
     */
    bool empty() const noexcept { return entries_.empty(); }

    /**
     * @brief Returns if it is a sequential dataset (for video).
     * @return True if the dataset is sequential, false otherwise.
     */
    bool is_sequential() const noexcept { return is_sequential_; }

    /**
     * @brief Sets whether the dataset is sequential.
     * @param val True to set the dataset as sequential, false otherwise.
     */
    void set_is_sequential(bool val) noexcept { is_sequential_ = val; }

    
    //const std::filesystem::path& get_path() const noexcept { return path_; }
    //const std::filesystem::path& get_image_root() const noexcept { return image_root_; }
    //const std::filesystem::path& get_annotation_root() const noexcept { return annotation_root_; }

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

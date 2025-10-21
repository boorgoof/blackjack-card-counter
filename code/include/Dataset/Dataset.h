#ifndef DATASET_H
#define DATASET_H

#include <cstddef>
#include <filesystem>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "../SampleInfo/SampleInfo.h"

/**
 * @brief Abstract base class for dataset management.
 * 
 * This class defines the interface for different dataset types (e.g., image folders, video streams).
 * Derived classes must implement the pure virtual methods to provide specific dataset functionality.
 */
class Dataset {
public:
    /**
     * A forward iterator for Dataset entries.
     * This iterator allows traversal over SampleInfo entries in the Dataset.
     * It supports standard iterator operations such as dereferencing, incrementing,
     * and comparison.
     */
    struct Iterator {
        using iterator_category = std::forward_iterator_tag;
        using difference_type   = std::ptrdiff_t;
        using value_type        = SampleInfo;
        using pointer           = SampleInfo*;
        using reference         = SampleInfo&;

    public:
        Iterator() = default;
        /**
         * Constructor for the Iterator.
         * @param current Iterator to the current SampleInfo entry.
         */
        explicit Iterator(std::vector<std::shared_ptr<SampleInfo>>::const_iterator current)
            : current_(current) {}

        /**
         * @brief Dereference operator to access the current SampleInfo.
         * @return Reference to the current SampleInfo.
         */
        reference operator*() const { return *(*current_); }

        /**
         * @brief Arrow operator to access members of the current SampleInfo.
         * @return Pointer to the current SampleInfo.
         */
        pointer operator->() const { return current_->get(); }

        /**
         * @brief Pre-increment operator to move to the next SampleInfo.
         * @return Reference to the incremented iterator.
         */
        Iterator& operator++() { ++current_; return *this; }

        /**
         * @brief Post-increment operator to move to the next SampleInfo.
         * @return Iterator before incrementing.
         */
        Iterator operator++(int) { Iterator tmp = *this; ++(*this); return tmp; }

        friend bool operator== (const Iterator& a, const Iterator& b) { return a.current_ == b.current_; };
        friend bool operator!= (const Iterator& a, const Iterator& b) { return a.current_ != b.current_; };
    private:
        std::vector<std::shared_ptr<SampleInfo>>::const_iterator current_;
    };

    /**
     * @brief Virtual destructor for proper cleanup of derived classes.
     */
    virtual ~Dataset() = default;

    /**
     * @brief Returns an iterator to the beginning of the Dataset entries.
     * @return Iterator to the first ImageInfo entry.
     */
    virtual Iterator begin() const = 0;

    /**
     * @brief Returns an iterator to the end of the Dataset entries.
     * @return Iterator to one past the last ImageInfo entry.
     */
    virtual Iterator end() const = 0;

    /**
     * @brief Returns the number of entries in the Dataset.
     * @return Size of the Dataset.
     */
    virtual size_t size() const noexcept = 0;

    /**
     * @brief Load the sample referenced by the iterator.
     * @param it Iterator pointing to a sample owned by this dataset.
     * @return Loaded cv::Mat image/frame.
     */
    virtual cv::Mat load(const Iterator& it) const = 0;

    /**
     * @brief Checks if the Dataset is empty.
     * @return True if the Dataset has no entries, false otherwise.
     */
    bool empty() const noexcept { return size() == 0; }

    /**
     * @brief Returns if it is a sequential dataset (for video).
     * @return True if the dataset is sequential, false otherwise.
     */
    virtual bool is_sequential() const noexcept = 0;

    /**
     * @brief Get the image root directory.
     * @return Path to the image directory, or empty path if not applicable.
     */
    virtual std::filesystem::path get_root() const = 0;

    /**
     * @brief Get the annotation root directory.
     * @return Path to the annotation directory, or empty path if not applicable.
     */
    virtual std::filesystem::path get_annotation_root() const = 0;

protected:
    /**
     * @brief Default constructor for derived classes.
     */
    Dataset() = default;

    /**
     * @brief Copy constructor (protected to prevent slicing).
     */
    Dataset(const Dataset&) = default;

    /**
     * @brief Copy assignment operator (protected to prevent slicing).
     */
    Dataset& operator=(const Dataset&) = default;

    /**
     * @brief Move constructor (protected to prevent slicing).
     */
    Dataset(Dataset&&) = default;

    /**
     * @brief Move assignment operator (protected to prevent slicing).
     */
    Dataset& operator=(Dataset&&) = default;
};

#endif 

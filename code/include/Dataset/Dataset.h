// Gianluca Caregnato

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
 * Derived classes (ImageDataset, VideoDataset, TemplateDataset)
 * implement the pure virtual methods for their specific source type.
 */
class Dataset {
public:
    /**
     * @brief Forward iterator over the SampleInfo entries of a Dataset.
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
         * @brief Construct from an underlying shared_ptr const_iterator.
         * @param current Iterator to the current SampleInfo entry.
         */
        explicit Iterator(std::vector<std::shared_ptr<SampleInfo>>::const_iterator current)
            : current_(current) {}

        reference operator*() const { return *(*current_); }
        pointer operator->() const { return current_->get(); }

        Iterator& operator++() { ++current_; return *this; }
        Iterator operator++(int) { Iterator tmp = *this; ++(*this); return tmp; }

        friend bool operator== (const Iterator& a, const Iterator& b) { return a.current_ == b.current_; };
        friend bool operator!= (const Iterator& a, const Iterator& b) { return a.current_ != b.current_; };
    private:
        std::vector<std::shared_ptr<SampleInfo>>::const_iterator current_;
    };

    virtual ~Dataset() = default;

    /**
     * @brief Returns an iterator to the beginning of the dataset.
     * @return Iterator to the first SampleInfo entry.
     */
    virtual Iterator begin() const = 0;

    /**
     * @brief Returns an iterator to the end of the dataset.
     * @return Iterator past the last SampleInfo entry.
     */
    virtual Iterator end() const = 0;

    /**
     * @brief Returns the number of entries in the dataset.
     * @return Number of samples.
     */
    virtual size_t size() const noexcept = 0;

    /**
     * @brief Load the sample referenced by the given iterator.
     * @param it Iterator pointing to a sample owned by this dataset.
     * @return The loaded cv::Mat image/frame.
     */
    virtual cv::Mat load(const Iterator& it) = 0;

    /**
     * @brief Checks if the dataset is empty.
     * @return True if the dataset has no entries.
     */
    bool empty() const noexcept { return size() == 0; }

    /**
     * @brief Returns whether the dataset must be consumed sequentially.
     * @return True if sequential (e.g. video), false otherwise.
     */
    virtual bool is_sequential() const noexcept = 0;

    /**
     * @brief Get the image/frame root directory.
     * @return Path to the source directory, or empty path if not applicable.
     */
    virtual std::filesystem::path get_root() const = 0;

    /**
     * @brief Get the annotation root directory.
     * @return Path to the annotation directory, or empty path if not applicable.
     */
    virtual std::filesystem::path get_annotation_root() const = 0;

    bool get_has_annotations() const { return has_annotations_; }
    void set_has_annotations(bool has_annotations) { has_annotations_ = has_annotations; }

protected:
    /** @brief True if the dataset has ground-truth annotations. */
    bool has_annotations_;

    Dataset(bool has_annotations = false) : has_annotations_{has_annotations} {};
    Dataset(const Dataset&) = default;
    Dataset& operator=(const Dataset&) = default;
    Dataset(Dataset&&) = default;
    Dataset& operator=(Dataset&&) = default;
};

#endif

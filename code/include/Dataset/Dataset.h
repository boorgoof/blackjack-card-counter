#ifndef DATASET_H
#define DATASET_H

#include <cstddef>
#include <filesystem>
#include <vector>
#include <string>

#include "../ImageInfo.h"

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
     * @brief Sets whether the dataset is sequential.
     * @param val True to set the dataset as sequential, false otherwise.
     */
    virtual void set_is_sequential(bool val) noexcept = 0;

    /**
     * @brief Get the image root directory.
     * @return Path to the image directory, or empty path if not applicable.
     */
    virtual std::filesystem::path get_image_root() const = 0;

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

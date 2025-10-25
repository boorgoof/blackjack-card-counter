#ifndef SAMPLEINFO_H
#define SAMPLEINFO_H

#include <string>

/**
 * @brief Base interface for dataset samples (images, video frames, etc.).
 *
 * Provides a minimal contract that allows Dataset iterators to expose
 * heterogeneous sample types through a common pointer.
 */
class SampleInfo {
public:
    SampleInfo() = default;
    virtual ~SampleInfo() = default;
    SampleInfo(const SampleInfo&) = delete;
    SampleInfo& operator=(const SampleInfo&) = delete;

    /**
     * @brief Returns whether the sample info is empty/invalid.
     */
    virtual bool empty() const noexcept = 0;

    /**
     * @brief Returns the logical name/id of the sample (without extension if applicable).
     */
    virtual const std::string& get_name() const noexcept = 0;

    /**
     * @brief Optional filesystem path to the underlying image frame.
     * Defaults to an empty string if the sample is not backed by a file on disk.
     */
    virtual const std::string& get_pathSample() const noexcept = 0;

    /**
     * @brief Optional filesystem path to the label/annotation, empty if none.
     */
    virtual const std::string& get_pathLabel() const noexcept = 0;

};

#endif // SAMPLEINFO_H

#ifndef VIDEODATASET_H
#define VIDEODATASET_H

#include "Dataset.h"
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/videoio.hpp>

#include "../SampleInfo/FrameInfo.h"


/**
 * @brief Concrete implementation of Dataset for video files.
 * 
 * This class manages datasets from video files, extracting frames at 1-second intervals.
 * Frames are decoded on-demand using cv::VideoCapture caching.
 */
class VideoDataset : public Dataset {
public:
    /**
     * @brief Construct from video file path.
     * @param video_path Path to the video file.
     * @param sample_fps Frames per second to sample from the video (default: 1.0).
     *                   For example, 10.0 means extract 10 frames per second.
     */
    VideoDataset(const std::string& video_path, double sample_fps = 1.0);
    ~VideoDataset() override = default;

    // Implement pure virtual methods from Dataset
    Iterator begin() const override { return Iterator(entries_.cbegin()); }
    Iterator end() const override { return Iterator(entries_.cend()); }
    size_t size() const noexcept override { return entries_.size(); }
    bool is_sequential() const noexcept override { return true; }
    std::filesystem::path get_root() const override { return video_root_; }
    std::filesystem::path get_annotation_root() const override { return {}; }
    cv::Mat load(const Iterator& it) override;

    /**
     * @brief Set the frame sampling rate and rebuild the dataset entries.
     * @param sample_fps Frames per second to sample from the video.
     */
    void setSampleFPS(double sample_fps);

private:
    /**
     * @brief Builds the dataset entries by analyzing the video file.
     * @param video_root Path to the video file.
     * @param frame_interval_seconds Interval in seconds between sampled frames (internal use).
     * @return A vector of SampleInfo objects representing frame entries.
     */
    static std::vector<std::shared_ptr<SampleInfo>> build_entries(const std::filesystem::path& video_root, double frame_interval_seconds);

    /**
     * @brief Appends frame entries from a video file to the entries vector.
     * @param video_file Path to the video file.
     * @param entries Vector to append the frame entries to.
     * @param frame_interval_seconds Interval in seconds between sampled frames (internal use).
     */
    static void append_frames(const std::filesystem::path& video_file, std::vector<std::shared_ptr<SampleInfo>>& entries, double frame_interval_seconds);

    /**
     * @brief State tracking for cached video captures.
     */
    struct CaptureState {
        cv::VideoCapture capture;
        std::size_t next_frame_index{0};
    };

    std::filesystem::path video_root_;
    // Interval in seconds between sampled frames (calculated as 1.0/sample_fps)
    // Must be before entries_ for proper initialization order
    double frame_interval_seconds_;
    std::vector<std::shared_ptr<SampleInfo>> entries_;
    // Cache to store VideoCapture objects for each video file
    std::unordered_map<std::string, CaptureState> capture_cache_;
};
#endif // VIDEODATASET_H

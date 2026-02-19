// Gianluca Caregnato

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
 * @brief Dataset implementation for video files.
 *
 * Extracts frames at a configurable sampling rate (default 1 fps).
 * Frames are decoded on-demand using cached cv::VideoCapture objects.
 */
class VideoDataset : public Dataset {
public:
    /**
     * @brief Construct from a video file path.
     * @param video_path   Path to the video file.
     * @param has_annotations Whether ground-truth annotations exist.
     * @param sample_fps   Frames per second to sample (e.g. 10.0 = 10 frames/s).
     */
    VideoDataset(const std::string& video_path, bool has_annotations = false, double sample_fps = 1.0);
    ~VideoDataset() override = default;

    Iterator begin() const override { return Iterator(entries_.cbegin()); }
    Iterator end() const override { return Iterator(entries_.cend()); }
    size_t size() const noexcept override { return entries_.size(); }
    bool is_sequential() const noexcept override { return true; }
    std::filesystem::path get_root() const override { return video_root_; }
    std::filesystem::path get_annotation_root() const override { return {}; }
    cv::Mat load(const Iterator& it) override;

    /**
     * @brief Change the sampling rate and rebuild entries.
     * @param sample_fps New frames-per-second value.
     */
    void setSampleFPS(double sample_fps);

    /**
     * @brief Get the current frame sampling interval.
     * @return Interval in seconds between sampled frames.
     */
    double getSampleFPS() const { return frame_interval_seconds_; }

private:
    /**
     * @brief Build all frame entries for the video.
     * @param video_root             Path to the video file.
     * @param frame_interval_seconds Interval in seconds between sampled frames.
     * @return Vector of SampleInfo (FrameInfo) shared pointers.
     */
    static std::vector<std::shared_ptr<SampleInfo>> build_entries(const std::filesystem::path& video_root, double frame_interval_seconds);

    /**
     * @brief Append sampled frame entries from a single video file.
     * @param video_file             Path to the video file.
     * @param entries                Vector to append entries to.
     * @param frame_interval_seconds Interval in seconds between sampled frames.
     */
    static void append_frames(const std::filesystem::path& video_file, std::vector<std::shared_ptr<SampleInfo>>& entries, double frame_interval_seconds);

    /** @brief Cached VideoCapture with read-head position tracking. */
    struct CaptureState {
        cv::VideoCapture capture;
        std::size_t next_frame_index{0};
    };

    std::filesystem::path video_root_;
    double frame_interval_seconds_;           ///< Must be declared before entries_ for init order.
    std::vector<std::shared_ptr<SampleInfo>> entries_;
    std::unordered_map<std::string, CaptureState> capture_cache_;
};

#endif // VIDEODATASET_H

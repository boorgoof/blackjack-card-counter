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
     */
    VideoDataset(const std::string& video_path);
    ~VideoDataset() override = default;

    // Implement pure virtual methods from Dataset
    Iterator begin() const override { return Iterator(entries_.cbegin()); }
    Iterator end() const override { return Iterator(entries_.cend()); }
    size_t size() const noexcept override { return entries_.size(); }
    bool is_sequential() const noexcept override { return true; }
    std::filesystem::path get_root() const override { return video_root_; }
    std::filesystem::path get_annotation_root() const override { return {}; }
    cv::Mat load(const Iterator& it) override;

private:
    /**
     * @brief Builds the dataset entries by analyzing the video file.
     * @param video_root Path to the video file.
     * @return A vector of SampleInfo objects representing frame entries.
     */
    static std::vector<std::shared_ptr<SampleInfo>> build_entries(const std::filesystem::path& video_root);

    /**
     * @brief Appends frame entries from a video file to the entries vector.
     * @param video_file Path to the video file.
     * @param entries Vector to append the frame entries to.
     */
    static void append_frames(const std::filesystem::path& video_file, std::vector<std::shared_ptr<SampleInfo>>& entries);

    /**
     * @brief State tracking for cached video captures.
     */
    struct CaptureState {
        cv::VideoCapture capture;
        std::size_t next_frame_index{0};
    };

    std::filesystem::path video_root_;
    std::vector<std::shared_ptr<SampleInfo>> entries_;
    // Cache to store VideoCapture objects for each video file
    std::unordered_map<std::string, CaptureState> capture_cache_;
};
#endif // VIDEODATASET_H

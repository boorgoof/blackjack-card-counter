#ifndef FRAMEINFO_H
#define FRAMEINFO_H

#include <cstddef>
#include <string>

#include "SampleInfo.h"

/**
 * @brief SampleInfo implementation representing a frame inside a video stream.
 *
 * Frames are decoded on demand via cv::VideoCapture to avoid pre-extracting them to disk.
 */
class FrameInfo : public SampleInfo {
public:
    FrameInfo() = default;
    FrameInfo(std::string name, std::string video_path, std::size_t frame_index, double timestamp_seconds = 0.0, std::string label_path = {})
        : name_{std::move(name)}, video_path_{std::move(video_path)}, frame_index_{frame_index}, timestamp_seconds_{timestamp_seconds}, label_path_{std::move(label_path)} {}

    bool empty() const noexcept override { return name_.empty(); }
    const std::string& get_name() const noexcept override { return name_; }
    const std::string& get_pathSample() const noexcept override { return video_path_; }
    const std::string& get_pathLabel() const noexcept override { return label_path_; }

    /**
     * @brief Returns the frame index within the video.
     */
    std::size_t get_frame_index() const noexcept { return frame_index_; }

    /**
     * @brief Returns the frame timestamp (seconds).
     */
    double get_timestamp() const noexcept { return timestamp_seconds_; }

private:
    std::string name_;
    std::string video_path_;
    std::size_t frame_index_{0};
    double timestamp_seconds_{0.0};
    std::string label_path_;
};

#endif // FRAMEINFO_H

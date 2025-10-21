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


class VideoDataset : public Dataset {
public:
    VideoDataset(const std::string& video_path);
    ~VideoDataset() override = default;

    // Implement pure virtual methods from Dataset
    Iterator begin() const override { return Iterator(entries_.cbegin()); }
    Iterator end() const override { return Iterator(entries_.cend()); }
    size_t size() const noexcept override { return entries_.size(); }
    bool is_sequential() const noexcept override { return true; }
    std::filesystem::path get_root() const override { return video_root_; }
    std::filesystem::path get_annotation_root() const override { return {}; }
    cv::Mat load(const Iterator& it) const override;

private:
    static std::vector<std::shared_ptr<SampleInfo>> build_entries(const std::filesystem::path& video_root);

    struct CaptureState {
        cv::VideoCapture capture;
        std::size_t next_frame_index{0};
    };

    std::filesystem::path video_root_;
    std::vector<std::shared_ptr<SampleInfo>> entries_;
    // Cache to store VideoCapture objects for each video file
    // putting it mutable allows that member to be modified even when the containing object is const
    mutable std::unordered_map<std::string, CaptureState> capture_cache_;
};
#endif // VIDEODATASET_H

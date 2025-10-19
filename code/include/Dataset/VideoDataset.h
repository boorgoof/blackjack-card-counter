#ifndef VIDEODATASET_H
#define VIDEODATASET_H

#include "Dataset.h"
#include <filesystem>
#include <string>
#include <vector>


class VideoDataset : public Dataset {
public:
    VideoDataset(const std::string& video_path);
    ~VideoDataset() override = default;

    // Implement pure virtual methods from Dataset
    Iterator begin() const override;
    Iterator end() const override;
    size_t size() const noexcept override { return entries_.size(); }
    bool is_sequential() const noexcept override { return true; }
    std::filesystem::path get_root() const override { return video_root_; }

private:
    std::filesystem::path video_root_;
    std::vector<VideoInfo> entries_;
};
#endif // VIDEODATASET_H
#include "../../include/Dataset/VideoDataset.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <memory>
#include <opencv2/videoio.hpp>

VideoDataset::VideoDataset(const std::string& video_path, bool has_annotations, double sample_fps)
    : Dataset(has_annotations), video_root_{video_path}, frame_interval_seconds_{1.0 / sample_fps}, entries_{build_entries(video_root_, frame_interval_seconds_)} { }


cv::Mat VideoDataset::load(const Dataset::Iterator& it) {
    if (entries_.empty() || it == Iterator(entries_.cend())) {
        return {};
    }

    const FrameInfo* frame_info = dynamic_cast<const FrameInfo*>(&*it);
    if (!frame_info) {
        std::cerr << "VideoDataset: iterator does not point to a FrameInfo object" << std::endl;
        return {};
    }

    const std::string& video_path = frame_info->get_pathSample();
    const std::size_t target_index = frame_info->get_frame_index();

    // Retrieve or create a cache entry for this video file
    std::pair<decltype(capture_cache_)::iterator, bool> cache_pair = capture_cache_.try_emplace(video_path); 
    decltype(capture_cache_)::iterator cache_it = cache_pair.first;
    bool inserted = cache_pair.second;
    CaptureState& state = cache_it->second;

    if (inserted || !state.capture.isOpened()) {
        if (!state.capture.open(video_path)) {
            std::cerr << "VideoDataset: unable to open video capture for " << video_path << std::endl;
            return {};
        }
        state.next_frame_index = 0;
    }

    // Seek only if the target is not the next sequential frame
    if (target_index != state.next_frame_index) {
        state.capture.set(cv::CAP_PROP_POS_FRAMES, static_cast<double>(target_index));
    }

    cv::Mat frame;
    if (!state.capture.read(frame)) {
        return {};
    }

    state.next_frame_index = target_index + 1;
    return frame;
}

std::vector<std::shared_ptr<SampleInfo>> VideoDataset::build_entries(const std::filesystem::path& video_root, double frame_interval_seconds) {
    std::vector<std::shared_ptr<SampleInfo>> entries;

    if (!std::filesystem::exists(video_root)) {
        std::cerr << "VideoDataset: video file does not exist: " << video_root << std::endl;
        return entries;
    }
    
    append_frames(video_root, entries, frame_interval_seconds);
    return entries;
}

void VideoDataset::append_frames(const std::filesystem::path& video_file, std::vector<std::shared_ptr<SampleInfo>>& entries, double frame_interval_seconds) {
    cv::VideoCapture capture(video_file.string());
    if (!capture.isOpened()) {
        std::cerr << "VideoDataset: unable to open video file " << video_file << std::endl;
        return;
    }

    const std::size_t frame_count = static_cast<std::size_t>(capture.get(cv::CAP_PROP_FRAME_COUNT));
    const double fps = capture.get(cv::CAP_PROP_FPS);
    const std::string video_name = video_file.stem().string();

    double duration_seconds = static_cast<double>(frame_count) / fps;
    // Subtract 2x interval as safety margin to avoid reading past the end
    double safe_duration = duration_seconds - (2.0 * frame_interval_seconds);
    std::size_t steps = static_cast<std::size_t>(std::floor(safe_duration / frame_interval_seconds)) + 1;
    
    entries.reserve(entries.size() + steps);
    
    for (std::size_t i = 0; i < steps && frame_count > 0; ++i) {
        double timestamp = static_cast<double>(i) * frame_interval_seconds;
        if (timestamp >= safe_duration) {
            break;
        }

        std::size_t frame_idx = 0;
        if (fps > 0.0) {
            frame_idx = static_cast<std::size_t>(std::llround(timestamp * fps));
            if (frame_idx >= frame_count) {
                break;
            }
        } else {
            frame_idx = std::min<std::size_t>(i, frame_count - 1);
            if (frame_idx >= frame_count - 1 && i > 0) {
                break;
            }
        }

        std::string name = video_name + "_t_" + std::to_string(timestamp); 
        entries.emplace_back(std::make_shared<FrameInfo>(name, video_file.string(), frame_idx, timestamp));
    }
}

void VideoDataset::setSampleFPS(double sample_fps) {
    frame_interval_seconds_ = 1.0 / sample_fps;
    capture_cache_.clear();
    entries_ = build_entries(video_root_, frame_interval_seconds_);
}

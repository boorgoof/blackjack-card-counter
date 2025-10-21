#include "../../include/Dataset/VideoDataset.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <memory>
#include <opencv2/videoio.hpp>

VideoDataset::VideoDataset(const std::string& video_path)
    : video_root_{video_path}, entries_{build_entries(video_root_)} { }


cv::Mat VideoDataset::load(const Dataset::Iterator& it) {
    // Check if the dataset is empty or iterator is at the end
    if (entries_.empty() || it == Iterator(entries_.cend())) {
        return {};
    }

    // Cast the iterator's SampleInfo to FrameInfo to access video-specific metadata
    const auto* frame_info = dynamic_cast<const FrameInfo*>(&*it);
    // Add null check
    if (!frame_info) {
        std::cerr << "VideoDataset: iterator does not point to a FrameInfo object" << std::endl;
        return {};
    }

    // Extract the video file path and the target frame index from the FrameInfo
    const std::string& video_path = frame_info->get_pathSample();
    const std::size_t target_index = frame_info->get_frame_index();

    // Try to retrieve or create a cache entry for this video file
    std::pair<decltype(capture_cache_)::iterator, bool> cache_pair = capture_cache_.try_emplace(video_path); 
    decltype(capture_cache_)::iterator cache_it = cache_pair.first;
    bool inserted = cache_pair.second; // true if a new entry was created
    CaptureState& state = cache_it->second;
    
    // If this is a new cache entry or the capture is not open, open the video file
    if (inserted || !state.capture.isOpened()) {
        if (!state.capture.open(video_path)) {
            std::cerr << "VideoDataset: unable to open video capture for " << video_path << std::endl;
            return {};
        }
        state.next_frame_index = 0; // Start at the beginning of the video
    }

    // If the target frame is not the next sequential frame, we need to seek
    if (target_index != state.next_frame_index) {
        state.capture.set(cv::CAP_PROP_POS_FRAMES, static_cast<double>(target_index)) ;
    }

    // Read the actual frame data into a cv::Mat
    cv::Mat frame;
    if (!state.capture.read(frame)) {
        std::cerr << "VideoDataset: failed to read frame " << target_index << " from " << video_path << std::endl;
        return {};
    }

    state.next_frame_index = target_index + 1;
    return frame;
}

std::vector<std::shared_ptr<SampleInfo>> VideoDataset::build_entries(const std::filesystem::path& video_root) {
    std::vector<std::shared_ptr<SampleInfo>> entries;

    if (!std::filesystem::exists(video_root)) {
        std::cerr << "VideoDataset: video file does not exist: " << video_root << std::endl;
        return entries;
    }
    
    append_frames(video_root, entries);
    return entries;
}

void VideoDataset::append_frames(const std::filesystem::path& video_file, std::vector<std::shared_ptr<SampleInfo>>& entries) {
    cv::VideoCapture capture(video_file.string());
    if (!capture.isOpened()) {
        std::cerr << "VideoDataset: unable to open video file " << video_file << std::endl;
        return;
    }

    const auto frame_count = static_cast<std::size_t>(capture.get(cv::CAP_PROP_FRAME_COUNT)); // Total number of frames
    const double fps = capture.get(cv::CAP_PROP_FPS); // Frames per second
    const std::string video_name = video_file.stem().string();

    double duration_seconds = static_cast<double>(frame_count) / fps;
    std::size_t steps = static_cast<std::size_t>(std::ceil(duration_seconds));
    
    entries.reserve(entries.size() + steps + 1);
    
    for (std::size_t second = 0; second <= steps && frame_count > 0; ++second) {
        double timestamp = static_cast<double>(second);
        std::size_t frame_idx = 0;
        if (fps > 0.0) {
            frame_idx = static_cast<std::size_t>(std::llround(timestamp * fps)); // Corresponding frame index
            if (frame_idx >= frame_count) {
                frame_idx = frame_count - 1;
            }
        } else {
            frame_idx = std::min<std::size_t>(second, frame_count - 1);
        }
        std::string name = video_name + "_t_" + std::to_string(second); // e.g., "video1_t_0", "video1_t_1"..
        entries.emplace_back(std::make_shared<FrameInfo>(name, video_file.string(), frame_idx, timestamp));
        if (frame_idx + 1 >= frame_count) {
            break;
        }
    }
}

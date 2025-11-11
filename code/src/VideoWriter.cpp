#include "../include/VideoWriter.h"
#include <filesystem>
#include <algorithm>

VideoWriter::VideoWriter(const std::string& outputPath, double fps)
    : outputPath_(outputPath), fps_(fps), initialized_(false) {}

VideoWriter::~VideoWriter() {
    close();
}

bool VideoWriter::isImageFile(const std::string& filename) {
    std::string lower = filename;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return lower.ends_with(".jpg") || lower.ends_with(".jpeg") || lower.ends_with(".png");
}

std::vector<std::string> VideoWriter::getImageFiles(const std::string& folderPath) {
    std::vector<std::string> imagePaths;
    
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(folderPath)) {
        if (entry.is_regular_file() && isImageFile(entry.path().filename().string())) {
            imagePaths.push_back(entry.path().string());
        }
    }
    
    std::sort(imagePaths.begin(), imagePaths.end());
    
    return imagePaths;
}

void VideoWriter::createVideoFromFolder(const std::string& folderPath) {
    std::vector<std::string> imagePaths = getImageFiles(folderPath);
    
    for (const std::string& path : imagePaths) {
        cv::Mat frame = cv::imread(path);
        if (!frame.empty()) {
            addFrame(frame);
        }
    }
}

void VideoWriter::addFrame(const cv::Mat& frame) {
    if (!initialized_) {
        frameSize_ = frame.size();
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        videoWriter_.open(outputPath_, fourcc, fps_, frameSize_);
        initialized_ = true;
    }
    
    if (frame.size() == frameSize_) {
        videoWriter_.write(frame);
    } else {
        cv::Mat resized;
        cv::resize(frame, resized, frameSize_);
        videoWriter_.write(resized);
    }
}

void VideoWriter::close() {
    if (videoWriter_.isOpened()) {
        videoWriter_.release();
    }
}

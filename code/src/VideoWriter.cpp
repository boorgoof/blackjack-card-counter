#include "../include/VideoWriter.h"
#include <filesystem>
#include <algorithm>

VideoWriter::VideoWriter(const std::string& outputPath, double fps)
    : outputPath(outputPath), fps(fps), initialized(false) {}

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
    
    std::filesystem::directory_iterator begin(folderPath);
    std::filesystem::directory_iterator end;
    
    for (std::filesystem::directory_iterator it = begin; it != end; ++it) {
        if (it->is_regular_file() && isImageFile(it->path().filename().string())) {
            imagePaths.push_back(it->path().string());
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
    if (!initialized) {
        frameSize = frame.size();
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        videoWriter.open(outputPath, fourcc, fps, frameSize);
        initialized = true;
    }
    
    cv::Mat resized;
    if (frame.size() != frameSize) {
        cv::resize(frame, resized, frameSize);
        videoWriter.write(resized);
    } else {
        videoWriter.write(frame);
    }
}

void VideoWriter::close() {
    if (videoWriter.isOpened()) {
        videoWriter.release();
    }
}

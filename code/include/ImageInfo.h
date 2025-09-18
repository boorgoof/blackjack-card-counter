#ifndef IMAGE_INFO_H
#define IMAGE_INFO_H

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "Label.h"

class ImageInfo {
public:
    ImageInfo() = default;
    ImageInfo(cv::Mat image, std::string path, std::vector<Label> labels)
        : image_{std::move(image)}, path_{std::move(path)}, labels_{std::move(labels)} {}

    bool empty() const noexcept { return image_.empty(); }
    int width() const noexcept { return image_.cols; }
    int height() const noexcept { return image_.rows; }

    const cv::Mat& image() const noexcept { return image_; }
    cv::Mat& image() noexcept { return image_; }

    const std::string& path() const noexcept { return path_; }

    const std::vector<Label>& labels() const noexcept { return labels_; }
    std::vector<Label>& labels() noexcept { return labels_; }

private:
    cv::Mat image_;
    std::string path_;
    std::vector<Label> labels_;
};

#endif

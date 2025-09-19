#ifndef IMAGE_INFO_H
#define IMAGE_INFO_H

#include <string>
#include <vector>
#include <utility>

#include <opencv2/opencv.hpp>

#include "Label.h"

class ImageInfo {
public:
    ImageInfo() = default;
    ImageInfo(std::string name, std::string image_path, std::string label_path)
        : name_{std::move(name)}, pathImage_{std::move(image_path)}, pathLabel_{std::move(label_path)} {}

    bool empty() const noexcept { return name_.empty(); }

    const std::string& name() const noexcept { return name_; }
    const std::string& path() const noexcept { return pathImage_; }
    const std::string& pathLabel() const noexcept { return pathLabel_; }

private:
    std::string name_;
    std::string pathImage_;
    std::string pathLabel_;
};

#endif

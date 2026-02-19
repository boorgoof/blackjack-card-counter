// Gianluca Caregnato

#ifndef IMAGE_INFO_H
#define IMAGE_INFO_H

#include <ostream>
#include <string>
#include <utility>

#include "SampleInfo.h"

/**
 * @brief SampleInfo implementation for a single image with its annotation path.
 */
class ImageInfo : public SampleInfo {
public:
    ImageInfo() = default;
    ImageInfo(std::string name, std::string image_path, std::string label_path)
        : name_{std::move(name)}, pathImage_{std::move(image_path)}, pathLabel_{std::move(label_path)} { }

    /**
     * @brief Check if the ImageInfo is empty.
     * @return True if the name is empty.
     */
    bool empty() const noexcept override { return name_.empty(); }

    /**
     * @brief Get the image name (without extension).
     * @return Name of the image.
     */
    const std::string& get_name() const noexcept override { return name_; }

    /**
     * @brief Get the path to the image file.
     * @return Path to the image file.
     */
    const std::string& get_pathSample() const noexcept override { return pathImage_; }

    /**
     * @brief Get the path to the label file.
     * @return Path to the label file.
     */
    const std::string& get_pathLabel() const noexcept override { return pathLabel_; }
    
    friend std::ostream& operator<<(std::ostream& os, const ImageInfo& info) {
        os << "ImageInfo{name: " << info.name_ << ", image_path: " << info.pathImage_ << ", label_path: " << info.pathLabel_ << "}";
        return os;
    }

private:
    std::string name_;
    std::string pathImage_;
    std::string pathLabel_;
};

#endif

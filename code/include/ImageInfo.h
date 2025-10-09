#ifndef IMAGE_INFO_H
#define IMAGE_INFO_H

#include <string>
#include <vector>
#include <utility>

#include <opencv2/opencv.hpp>

#include "Label.h"
#include "Loaders.h"

/**
 * @class ImageInfo
 * @brief Class representing information about an image and its associated label.
 */
class ImageInfo {
public:
    ImageInfo() = default;
    ImageInfo(std::string name, std::string image_path, std::string label_path)
        : name_{std::move(name)}, pathImage_{std::move(image_path)}, pathLabel_{std::move(label_path)} {}

    /**
     * @brief Checks if the ImageInfo is empty.
     * @return True if the ImageInfo is empty, false otherwise.
     */
    bool empty() const noexcept { return name_.empty(); }


    /**
     * @brief get the name of the image (without extension).
     * @return Name of the image.
     */
    const std::string& get_name() const noexcept { return name_; }

    /**
     * @brief get the path to the image file.
     * @return Path to the image file.
     */
    const std::string& get_pathImage() const noexcept { return pathImage_; }

    /**
     * @brief get the path to the label file.
     * @return Path to the label file.
     */
    const std::string& get_pathLabel() const noexcept { return pathLabel_; }
    
    /**
     * @brief Load the image.
     * @return Loaded cv::Mat image.
     */
    cv::Mat load_image() const { return Loader::Image::load_image(pathImage_); }
    /**
     * @brief Overload the output stream operator for ImageInfo.
     * @param os Output stream.
     * @param info ImageInfo object to output.
     * @return Reference to the output stream.
     */
    friend std::ostream& operator<<(std::ostream& os, const ImageInfo& info) {
        os << "ImageInfo{name: " << info.name_ << ", image_path: " << info.pathImage_ << ", label_path: " << info.pathLabel_ << "}";
        return os;
    }

private:
    // Image name (without extension)
    std::string name_;
    // Path to the image file
    std::string pathImage_;
    // Path to the label file
    std::string pathLabel_;
};

#endif

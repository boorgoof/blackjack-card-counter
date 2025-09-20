//Matteo Bino

#ifndef IMAGEFILTER_H
#define IMAGEFILTER_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <string>
#include <functional>
#include <utility>

/**
 * @brief class to apply a series of image filters to an image.
 */
class ImageFilter{
    private:
    /**
     * @brief A vector of pairs, where each pair contains a filter name and a function that takes an image as input and returns a filtered image.
     */
    std::vector<std::pair<std::string, std::function<cv::Mat(cv::Mat)>>> filter_pipeline;

    public:
    ImageFilter(){}
    ~ImageFilter();
    /**
     * @brief Applies the filter pipeline to the input image.
     * @param src_img The input image to be filtered.
     * @return The filtered image.
     */
    cv::Mat apply_filters(const cv::Mat& src_img) const;

    /**
     * @brief Adds a filter to the filter pipeline.
     * @param filter_name The name of the filter.
     * @param filter_function The function that implements the filter.
     * @param args The arguments to be passed to the filter function.
     * @note suitable filter functions are in the Filters namespace.
     */
    template<typename FilterFunction, typename... Args>
    void add_filter(const std::string& filter_name, FilterFunction filter_function, Args&&... args);
    /**
     * @brief Removes a filter from the filter pipeline.
     * @param filter_name The name of the filter to be removed.
     * @return true if the filter was removed, false otherwise.
     */
    bool remove_filter(const std::string& filter_name);

    const std::vector<std::pair<std::string, std::function<cv::Mat(cv::Mat)>>> get_filters() const {
        return this->filter_pipeline;
    }
};
/**
 * @brief A namespace containing various image filtering functions.
 */
namespace Filters{

    cv::Mat gaussian_blur(const cv::Mat& src_img, const cv::Size& kernel_size);
    cv::Mat median_blur(const cv::Mat& src_img, const cv::Size& kernel_size);
    cv::Mat average_blur(const cv::Mat& src_img, const cv::Size& kernel_size);
    /**
     * @brief Applies a bilateral filter to the input image.
     * @param src_img The input image to be filtered.
     * @param diameter Diameter of the pixel neighborhood.
     * @param sigma_color Filter sigma in color space.
     * @param sigma_space Filter sigma in coordinate space.
     * @return The filtered image.
     * @note The function converts the image to LAB color space, applies the bilateral filter to the L channel (luminance), and then converts back to BGR color space.
     */
    cv::Mat bilateral_filter(const cv::Mat& src_img, int diameter, double sigma_color, double sigma_space);
    /**
     * @brief Applies global contrast equalization to the input image.
     * @param src_img The input image to be filtered.
     * @return The filtered image.
     * @note The function converts the image to LAB color space, equalizes the histogram of the L channel (luminance), and then converts back to BGR color space.
     */
    cv::Mat global_contrast_equalization(const cv::Mat& src_img);
    /**
     * @brief Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to the input image.
     * @param src_img The input image to be filtered.
     * @param clip_limit The threshold for contrast limiting.
     * @param tile_grid_size The size of the grid for histogram equalization.
     * @return The filtered image.
     * @note The function converts the image to LAB color space, applies CLAHE to the L channel (luminance), and then converts back to BGR color space.
     */
    cv::Mat CLAHE_contrast_equalization(const cv::Mat& src_img, int clip_limit, int tile_grid_size);
    /**
     * @brief Applies unsharp masking to the input image.
     * @param src_img The input image to be filtered.
     * @param sigma The standard deviation of the Gaussian kernel.
     * @param alpha The weight of the sharpened image.
     * @return The filtered image.
     * @note The function converts the image to LAB color space, applies Gaussian blur to the L channel (luminance), and then combines the original and blurred images to create a sharpened effect.
     */
    cv::Mat unsharp_mask(const cv::Mat& src_img, double sigma, double alpha);

    cv::Mat resize(const cv::Mat& src_img, const float width_mult, const float height_mult);
}

template<typename FilterFunction, typename... Args>
void ImageFilter::add_filter(const std::string& filter_name, FilterFunction filter_function, Args&&... args){
    auto packaged_function = [filter_function, args...](const cv::Mat& img) {
        return filter_function(img, args...);
    };
    this->filter_pipeline.push_back(std::make_pair(filter_name, packaged_function));
}

#endif
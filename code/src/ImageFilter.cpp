//Matteo Bino

#include "../include/ImageFilter.h"

ImageFilter::~ImageFilter(){
    this->filter_pipeline.clear();
}

cv::Mat ImageFilter::apply_filters(const cv::Mat& src_img) const {   
    cv::Mat filtered_img = src_img.clone();
    for(auto& filter : this->filter_pipeline){
        //std::cout << "Applying filter: " << filter.first << std::endl;
        filtered_img = filter.second(filtered_img);
    }
    return filtered_img;
}

bool ImageFilter::remove_filter(const std::string& filter_name){
    this->filter_pipeline.erase(std::remove_if(this->filter_pipeline.begin(), this->filter_pipeline.end(),
        [&filter_name](const std::pair<std::string, std::function<cv::Mat(cv::Mat)>>& filter) {
            return filter.first == filter_name;
        }), this->filter_pipeline.end());

    return false;
}

cv::Mat Filters::gaussian_blur(const cv::Mat& src_img, const cv::Size& kernel_size){
    if (kernel_size.width <= 0 || kernel_size.height <= 0 || kernel_size.width % 2 == 0 || kernel_size.height % 2 == 0) {
        throw std::runtime_error("Gaussian kernel dimensions must be positive and odd: " + std::to_string(kernel_size.width) + "x" + std::to_string(kernel_size.height));
    }
    cv::Mat dst_img;
    cv::GaussianBlur(src_img, dst_img, kernel_size, 0);
    return dst_img;
}
cv::Mat Filters::median_blur(const cv::Mat& src_img, const cv::Size& kernel_size){
    if (kernel_size.width <= 0 || kernel_size.height <= 0 || kernel_size.width % 2 == 0 || kernel_size.height % 2 == 0) {
        throw std::runtime_error("Median kernel dimensions must be positive and odd: " + std::to_string(kernel_size.width) + "x" + std::to_string(kernel_size.height));
    }
    cv::Mat dst_img;
    cv::medianBlur(src_img, dst_img, kernel_size.width);
    return dst_img;
}
cv::Mat Filters::average_blur(const cv::Mat& src_img, const cv::Size& kernel_size){
    if (kernel_size.width <= 0 || kernel_size.height <= 0 || kernel_size.width % 2 == 0 || kernel_size.height % 2 == 0) {
        throw std::runtime_error("Average kernel dimensions must be positive and odd: " + std::to_string(kernel_size.width) + "x" + std::to_string(kernel_size.height));
    }
    cv::Mat dst_img;
    cv::blur(src_img, dst_img, kernel_size);
    return dst_img;
}
cv::Mat Filters::bilateral_filter(const cv::Mat &src_img, int diameter, double sigma_color, double sigma_space)
{
    //convert the image to LAB color space
    cv::Mat lab_img;
    cv::cvtColor(src_img, lab_img, cv::COLOR_BGR2Lab);
    //apply bilateral filter to the L channel (luminance)
    std::vector<cv::Mat> lab_channels;
    cv::split(lab_img, lab_channels);
    cv::Mat filtered_y_channel;
    cv::bilateralFilter(lab_channels[0], filtered_y_channel, diameter, sigma_color, sigma_space);
    //merge the filtered L channel with the original other channels
    lab_channels[0] = filtered_y_channel;
    cv::Mat filtered_lab_img;
    cv::merge(lab_channels, filtered_lab_img);
    //convert back to BGR color space
    cv::Mat filtered_bgr_img;
    cv::cvtColor(filtered_lab_img, filtered_bgr_img, cv::COLOR_Lab2BGR);
    return filtered_bgr_img;
}
cv::Mat Filters::global_contrast_equalization(const cv::Mat &src_img)
{
    cv::Mat dst_img;
    //converts the image from BGR to LAB
    cv::cvtColor(src_img, dst_img, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> lab_channels;
    //splits the LAB image into its channels
    cv::split(dst_img, lab_channels);
    //equalizes the histogram of the L channel (luminance)
    cv::equalizeHist(lab_channels[0], lab_channels[0]);
    cv::merge(lab_channels, dst_img);
    //after merging the channels, convert back to BGR
    cv::cvtColor(dst_img, dst_img, cv::COLOR_Lab2BGR);
    return dst_img;
}

cv::Mat Filters::CLAHE_contrast_equalization(const cv::Mat &src_img, int clip_limit, int tile_grid_size)
{
    cv::Mat dst_img;
    //convert the image from BGR to LAB 
    cv::cvtColor(src_img, dst_img, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> lab_channels;
    //split the LAB image into its channels
    cv::split(dst_img, lab_channels);
    //create a CLAHE object
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(clip_limit);
    clahe->setTilesGridSize(cv::Size(tile_grid_size, tile_grid_size));
    //apply CLAHE to the L channel (luminance)
    clahe->apply(lab_channels[0], lab_channels[0]);
    cv::merge(lab_channels, dst_img);
    //after merging the channels, convert back to BGR
    cv::cvtColor(dst_img, dst_img, cv::COLOR_Lab2BGR);
    return dst_img;
}
cv::Mat Filters::unsharp_mask(const cv::Mat &src_img, double sigma, double alpha)
{
    //convert the image to LAB color space
    cv::Mat lab_img;
    cv::cvtColor(src_img, lab_img, cv::COLOR_BGR2Lab);
    //apply Gaussian blur to the L channel (luminance)
    std::vector<cv::Mat> lab_channels;
    cv::split(lab_img, lab_channels);
    // Create a Gaussian kernel
    int kernel_size = static_cast<int>(sigma * 3) | 1; // Ensure kernel size is odd
    cv::Mat gaussian_kernel = cv::getGaussianKernel(kernel_size, sigma);
    cv::Mat blurred_img;
    cv::filter2D(lab_channels[0], blurred_img, -1, gaussian_kernel * gaussian_kernel.t());

    // Create the unsharp mask
    cv::Mat unsharp_mask = lab_channels[0] - blurred_img;

    // Add the unsharp mask to the original image
    cv::Mat sharpened_channel = lab_channels[0] + alpha * unsharp_mask;

    //merge channels
    lab_channels[0] = sharpened_channel;
    cv::Mat sharpened_lab_img;
    cv::merge(lab_channels, sharpened_lab_img);
    //convert back to BGR color space
    cv::Mat sharpened_img;
    cv::cvtColor(sharpened_lab_img, sharpened_img, cv::COLOR_Lab2BGR);

    return sharpened_img;
}

cv::Mat Filters::resize(const cv::Mat& src_img, const float width_mult, const float height_mult){
    if (width_mult <= 0 || height_mult <= 0) {
        throw std::runtime_error("Width and height multipliers must be positive: " + std::to_string(width_mult) + ", " + std::to_string(height_mult));
    }
    cv::Mat dst_img;
    cv::resize(src_img, dst_img, cv::Size(), width_mult, height_mult);
    return dst_img;
}
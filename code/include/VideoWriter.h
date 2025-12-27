#ifndef VIDEOWRITER_H
#define VIDEOWRITER_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

/**
 * @class VideoWriter
 * @brief A class to create MP4 videos from a folder of images
 */
class VideoWriter {
public:
    /**
     * @brief Constructor
     * @param outputPath Path where the output video will be saved
     * @param fps Frames per second for the output video (default: 30)
     */
    VideoWriter(const std::string& outputPath, double fps = 1.0);
    
    /**
     * @brief Destructor
     */
    ~VideoWriter();
    
    /**
     * @brief Create a video from all images in a folder
     * @param folderPath Path to the folder containing images
     */
    void createVideoFromFolder(const std::string& folderPath);
    
    /**
     * @brief Add a single frame to the video
     * @param frame The image frame to add
     */
    void addFrame(const cv::Mat& frame);
    
    /**
     * @brief Finalize and close the video file
     */
    void close();
    
private:
    /**
     * @brief Get all image files from a folder
     * @param folderPath Path to the folder
     * @return Vector of image file paths
     */
    std::vector<std::string> getImageFiles(const std::string& folderPath);
    
    /**
     * @brief Check if a file has a valid image extension
     * @param filename The filename to check
     * @return true if it's a valid image file, false otherwise
     */
    bool isImageFile(const std::string& filename);
    
    std::string outputPath_;
    double fps_;
    cv::VideoWriter videoWriter_;
    cv::Size frameSize_;
    bool initialized_;
};

#endif // VIDEOWRITER_H

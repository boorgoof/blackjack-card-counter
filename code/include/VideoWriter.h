// Gianluca Caregnato

#ifndef VIDEOWRITER_H
#define VIDEOWRITER_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

/**
 * @brief Creates MP4 videos from individual frames or a folder of images.
 */
class VideoWriter {
public:
    /**
     * @brief Construct a VideoWriter.
     * @param outputPath Path where the output video will be saved.
     * @param fps        Frames per second for the output video.
     */
    VideoWriter(const std::string& outputPath, double fps = 1.0);
    ~VideoWriter();
    
    /**
     * @brief Create a video from all images in a folder.
     * @param folderPath Path to the folder containing images.
     */
    void createVideoFromFolder(const std::string& folderPath);
    
    /**
     * @brief Add a single frame to the video.
     * @param frame The image frame to add.
     */
    void addFrame(const cv::Mat& frame);
    
    /**
     * @brief Finalize and close the video file.
     */
    void close();

    const std::string& get_output_path() const { return outputPath_; }
    void set_output_path(const std::string& outputPath) { outputPath_ = outputPath; }
    
private:
    /**
     * @brief Collect sorted image file paths from a folder.
     * @param folderPath Path to the folder.
     * @return Sorted vector of image file paths.
     */
    std::vector<std::string> getImageFiles(const std::string& folderPath);
    
    /**
     * @brief Check if a filename has a valid image extension (.jpg, .jpeg, .png).
     * @param filename The filename to check.
     * @return True if it has a valid image extension.
     */
    bool isImageFile(const std::string& filename);
    
    std::string outputPath_;
    double fps_;
    cv::VideoWriter videoWriter_;
    cv::Size frameSize_;
    bool initialized_;
};

#endif // VIDEOWRITER_H

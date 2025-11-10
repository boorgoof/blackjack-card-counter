#include <iostream>
#include <cstddef>
#include <opencv2/opencv.hpp>
#include "../include/card_detector/RoughCardDetector.h"
#include "../include/Dataset/Dataset.h"
#include "../include/Dataset/ImageDataset.h"
#include "../include/Dataset/VideoDataset.h"
#include "../include/SampleInfo/SampleInfo.h"
#include "../include/VideoWriter.h"
#include <memory>
#include <vector>

int main() {
    VideoWriter writer("data/output_video.mp4", 1.0);
    writer.createVideoFromFolder("data/datasets/videos/images");
    writer.close();
    
    std::cout << "Video created successfully!" << std::endl;
    
    return 0;
}

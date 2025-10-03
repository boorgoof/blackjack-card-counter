#ifndef OBJECT_DETECTOR_H
#define OBJECT_DETECTOR_H

#include <opencv2/opencv.hpp>
#include "../../Label.h"
#include "../../Dataset.h"


/**
 * @brief ObjectDetector class to detect objects in images.
 *        This is an abstract class that defines the interface for all object detectors.
 */
class ObjectDetector{

    public:

        ObjectDetector() = default;

        ObjectDetector(ObjectDetector&&) = delete;
        ObjectDetector& operator=(ObjectDetector&&) = delete; //TODO
        virtual ~ObjectDetector() = 0;

        
        /**
         * @brief detect objects in a scene image
         * @param src_img the scene image to detect objects from
         * @param src_mask the mask of the scene image (the area of interest where to search for objects)
         * @param out_labels the output vector of detected labels
         */
        virtual void detect_objects(const cv::Mat& src_img,  const cv::Mat &src_mask, std::vector<Label>& out_labels) = 0;

        void set_method_name(const std::string& method_name) { this->method_name = method_name; }
        const std::string& get_method_name() const { return this->method_name; }
        
    private:
    
        std::string method_name;

};

#endif // OBJECT_DETECTOR_H
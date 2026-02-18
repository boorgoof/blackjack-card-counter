//Federico Meneghetti

#ifndef OBJECT_CLASSIFIER_H
#define OBJECT_CLASSIFIER_H

#include <opencv2/opencv.hpp>
#include "../../../Label.h"
#include "../../../Dataset/Dataset.h"


/**
 * @brief ObjectClassifier class to detect objects in images.
 *        This is an abstract class that defines the interface for all object detectors.
 */
class ObjectClassifier{

public:

    ObjectClassifier() = default;

    ObjectClassifier(ObjectClassifier&&) = delete;
    ObjectClassifier& operator=(ObjectClassifier&&) = delete; 
    virtual ~ObjectClassifier() = 0;

    /**
     * @brief classify an object in an image given a mask. It is virtual and should be implemented by the specific object classifier classes (e.g. FeaturePipeline)
     */
    virtual const ObjectType* classify_object(const cv::Mat& src_img,  const cv::Mat &src_mask) = 0;

    /**
     * @brief set the name of the classification method
     * @param method_name the name of the classification method to set
     */
    void set_method_name(const std::string& method_name) { this->method_name = method_name; }

    /**
     * @brief get the name of the classification method
     * @return the name of the classification method
     */
    const std::string& get_method_name() const { return this->method_name; }
    
private:
    std::string method_name;
};

#endif // OBJECT_CLASSIFIER_H
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

    virtual const ObjectType* classify_object(const cv::Mat& src_img,  const cv::Mat &src_mask) = 0;

    void set_method_name(const std::string& method_name) { this->method_name = method_name; }
    const std::string& get_method_name() const { return this->method_name; }
    
private:
    std::string method_name;
};

#endif // OBJECT_CLASSIFIER_H
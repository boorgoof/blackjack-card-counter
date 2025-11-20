#ifndef LABEL_H
#define LABEL_H

#include <opencv2/opencv.hpp>
# include "ObjectType.h"
class Label {
private:

    std::unique_ptr<ObjectType> object_;
    std::vector<cv::Rect> bounding_boxes_;
   

public:

    Label(): object_{nullptr},  bounding_boxes_{} {}

    Label(std::unique_ptr<ObjectType> obj, const cv::Rect& bbox, float conf = 0.f)
        : object_{std::move(obj)}, bounding_boxes_{bbox} {}

    Label(std::unique_ptr<ObjectType> obj, const std::vector<cv::Rect>& bboxes,float conf = 0.f)
        : object_{std::move(obj)}, bounding_boxes_{bboxes} {}

    Label(Label&& label) : object_{std::move(label.object_)}, bounding_boxes_{label.bounding_boxes_} {}
    Label& operator=(const Label& label) {
        object_ = label.object_ ? label.object_->clone() : nullptr;
        bounding_boxes_ = label.bounding_boxes_;
        return *this;
    }

    // getters
    const ObjectType* get_object() const { return object_.get(); }
    const std::vector<cv::Rect>& get_bounding_boxes() const { return bounding_boxes_;}

    // setters
    void set_object(std::unique_ptr<ObjectType> obj) { object_ = std::move(obj); }
    void set_bounding_boxes(const std::vector<cv::Rect>& bbox) { bounding_boxes_ = bbox; }
   
};

inline std::ostream& operator<<(std::ostream& os, const Label& l) {

    os << l.get_object() << " ";

    const auto& boxes = l.get_bounding_boxes();
    os << "[";
    for (size_t i = 0; i < boxes.size(); ++i) {
        os << boxes[i];
        if (i + 1 < boxes.size()) {
            os << ", ";
        }
    }
    os << "]";

    return os;
}

#endif // LABEL_H
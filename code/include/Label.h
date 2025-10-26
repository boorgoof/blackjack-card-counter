#ifndef LABEL_H
#define LABEL_H

#include "CardType.h"
#include <opencv2/opencv.hpp>

class Label {
private:

    std::unique_ptr<ObjectType> object_;
    cv::Rect boundingbox_;
    float confidence_{0.f};     

public:

    Label(): object_{nullptr}, boundingbox_{}, confidence_{0.f} {}

    Label(std::unique_ptr<ObjectType> obj, const cv::Rect& bbox, float conf = 0.f)
        : object_{std::move(obj)}, boundingbox_{bbox}, confidence_{conf} {}

    Label(Label&& label) : object_{std::move(label.object_)}, boundingbox_{label.boundingbox_}, confidence_{label.confidence_} {}
    Label& operator=(const Label& label) {
        object_ = label.object_ ? label.object_->clone() : nullptr;
        boundingbox_ = label.boundingbox_;
        confidence_ = label.confidence_;
        return *this;
    }

    // getters
    const ObjectType* get_object() const { return object_.get(); }
    const cv::Rect& get_bounding_box() const { return boundingbox_;}
    float get_confidence() const { return confidence_;}  

    // setters
    void set_object(std::unique_ptr<ObjectType> obj) { object_ = std::move(obj); }
    void set_bounding_box(const cv::Rect& bbox) { boundingbox_ = bbox; }
    void set_confidence(float conf) { confidence_ = conf; }      
};

inline std::ostream& operator<<(std::ostream& os, const Label& l){
    os << l.get_object() << " " << l.get_bounding_box();
    return os;
}

#endif // LABEL_H
#ifndef LABEL_H
#define LABEL_H

#include "CardType.h"

class Label {

    private:
        Card_Type class_name_;
        cv::Rect  boundingbox_;
    
    public:

        Label(const Card_Type& card, const cv::Rect& bbox, float conf = 0.f)
            : class_name_{card}, boundingbox_{bbox} {}

        
        const Card_Type& get_class_name() const noexcept { return class_name_; }
        const cv::Rect&  get_bounding_box() const noexcept { return boundingbox_; }

        void set_class_name(const Card_Type& new_name) { class_name_ = new_name; }
        void set_bounding_box(const cv::Rect& new_bb) { boundingbox_ = new_bb; }
        
};

inline std::ostream& operator<<(std::ostream& os, const Label& l){
    os << l.get_class_name() << " " << l.get_bounding_box();
    return os;
}

#endif // LABEL_H
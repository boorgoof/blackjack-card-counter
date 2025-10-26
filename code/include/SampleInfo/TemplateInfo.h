#ifndef TEMPLATE_INFO_H
#define TEMPLATE_INFO_H

#include "SampleInfo.h"
#include "../CardType.h"
#include <string>
#include <ostream>

class TemplateInfo : public SampleInfo {

public:
    TemplateInfo(const std::string& name, const std::string& image_path, const CardType& card_type)
        : name_{name}, pathSample_{image_path}, card_type_{card_type} { }

    bool empty() const noexcept override {
        return name_.empty() || pathSample_.empty() || !card_type_.isValid();
    }

    const std::string& get_name() const noexcept override {
        return name_;
    }

    const std::string& get_pathSample() const noexcept override {
        return pathSample_;
    }
    
    const std::string& get_pathLabel() const noexcept override {
        static const std::string kEmpty{};
        return kEmpty;
    }

    const CardType& get_card_type() const noexcept {
        return card_type_;
    }
    
    friend std::ostream& operator<<(std::ostream& os, const TemplateInfo& info) {
        os << "TemplateInfo{name: " << info.name_ << ", image_path: " << info.pathSample_ 
           << ", card_type: " << info.card_type_ << "}";
        return os;
    }

private:
    std::string name_;
    std::string pathSample_;
    CardType card_type_;

};

#endif // TEMPLATE_INFO_H

#ifndef FEATURE_H
#define FEATURE_H


class Feature {
    public:
        Feature() = default;
        Feature(const Feature&) = delete;
        Feature& operator=(const Feature&) = delete;
        
        virtual ~Feature() = default;


};

#endif // FEATURE_H
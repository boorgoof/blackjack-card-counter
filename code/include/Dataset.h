#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>
#include <stdexcept>

class Dataset {
private:
    std::vector<std::string> data;
    size_t current_index = 0; // used for next()

public:
    // constructors
    Dataset() = default;
    Dataset(std::initializer_list<std::string> init);

    // modifiers
    void add(const std::string& value);

    // accessors
    const std::string& next();
    const std::string& at(size_t index) const;
    const std::string& operator[](size_t index) const;

    // iteration support (if you want to use range-for)
    using iterator = std::vector<std::string>::iterator;
    using const_iterator = std::vector<std::string>::const_iterator;

    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;

    // utility
    size_t size() const;
    void reset();
};

#endif // DATASET_H

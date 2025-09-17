#include "Dataset.h"

// constructor
Dataset::Dataset(std::initializer_list<std::string> init) : data(init) {}

// add new element
void Dataset::add(const std::string& value) {
    data.push_back(value);
}

// return next element (wraps around at the end)
const std::string& Dataset::next() {
    if (data.empty()) {
        throw std::out_of_range("Dataset is empty");
    }
    const std::string& item = data[current_index];
    current_index = (current_index + 1) % data.size(); // circular
    return item;
}

// safe access
const std::string& Dataset::at(size_t index) const {
    return data.at(index); // throws std::out_of_range if invalid
}

// direct access
const std::string& Dataset::operator[](size_t index) const {
    return data[index]; // no bound check
}

// iteration support
Dataset::iterator Dataset::begin() { return data.begin(); }
Dataset::iterator Dataset::end() { return data.end(); }
Dataset::const_iterator Dataset::begin() const { return data.begin(); }
Dataset::const_iterator Dataset::end() const { return data.end(); }

// utility
size_t Dataset::size() const {
    return data.size();
}

void Dataset::reset() {
    current_index = 0;
}

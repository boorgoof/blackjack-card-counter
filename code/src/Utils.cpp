#include "../include/Utils.h"



std::string Utils::String::normalize(const std::string& str) {
    
    std::string t = str;  
    
    t.erase(std::remove_if(t.begin(), t.end(), [](unsigned char c){ return std::isspace(c); }), t.end());
    std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c){ return std::toupper(c); });
    
    return t;
}
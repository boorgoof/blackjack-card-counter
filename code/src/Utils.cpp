#include "../include/Utils.h"

std::string Utils::Path::longestCommonPath(const std::string& path1_str, const std::string& path2_str) {
    std::filesystem::path path1(path1_str);
    std::filesystem::path path2(path2_str);

    return longestCommonPath(path1, path2).string();
}

std::filesystem::path Utils::Path::longestCommonPath(const std::filesystem::path& path1, const std::filesystem::path& path2) {
    auto it1 = path1.begin();
    auto it2 = path2.begin();
    std::filesystem::path common_path;

    while (it1 != path1.end() && it2 != path2.end() && *it1 == *it2) {
        common_path /= *it1;
        ++it1;
        ++it2;
    }

    return common_path;
}

std::string Utils::String::normalize(const std::string& str) {
    
    std::string t = str;  
    
    t.erase(std::remove_if(t.begin(), t.end(), [](unsigned char c){ return std::isspace(c); }), t.end());
    std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c){ return std::toupper(c); });
    
    return t;
}

void Utils::Visualization::printProgressBar(float progress, size_t barwidth, const std::string& prefix, const std::string& suffix) {
    std::cout << "\r" << prefix << " [";
    size_t pos = static_cast<size_t>(barwidth * progress);
    for (size_t i = 0; i < barwidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% " << suffix << std::flush;
    if (progress >= 1.0) std::cout << std::endl;
}
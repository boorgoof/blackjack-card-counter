#ifndef UTILS_H
#define UTILS_H

#include <string>
#include "Label.h"
#include <map>
#include <filesystem>

/**
 * @brief utility functions.
 */
namespace Utils{

    namespace String{
        std::string normalize(const std::string& str); 
    }

    namespace Path{
        std::string longestCommonPath(const std::string& path1_str, const std::string& path2_str);
        std::filesystem::path longestCommonPath(const std::filesystem::path& path1, const std::filesystem::path& path2);
    }
    
    namespace Visualization{
        void printProgressBar(float progress, size_t barwidth, const std::string& prefix = "", const std::string& suffix = "");
    }

    /**
     * @brief functions to handle maps.
     */
    namespace Map{
        /**
         * @brief function to create an inverse map from a given map.
         * @tparam MapA2B the type of the map to be inverted
         * @tparam MapB2A the type of the inverted map
         * @param map the map to be inverted
         * @return the inverted map
         * 
         * @note function gently retrieved from //https://stackoverflow.com/questions/54398336/stl-type-for-mapping-one-to-one-relations
         */
        template <typename MapA2B, typename MapB2A = std::map<typename MapA2B::mapped_type, typename MapA2B::key_type>>
        MapB2A createInverseMap(const MapA2B& map){
            MapB2A inverseMap;
            for (const auto& pair : map) {
                inverseMap.emplace(pair.second, pair.first);
            }
            return inverseMap;
        }
    }

};

#endif
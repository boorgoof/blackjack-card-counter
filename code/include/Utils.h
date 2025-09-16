#ifndef UTILS_H
#define UTILS_H

#include <string>
#include "Label.h"
#include <map>

/**
 * @brief utility functions.
 */
namespace Utils{


    namespace String{

        std::string normalize(std::string_view str) {

            std::string out; 
            out.reserve(str.size());

            for (unsigned char c : str) {
                if (!std::isspace(c) && c != '_')
                    out.push_back(static_cast<char>(std::toupper(c)));
            }
            return out;
        }

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
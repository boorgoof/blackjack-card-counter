cmake_minimum_required(VERSION 3.10)
project(EasyPeasyCV-Blackjack)

find_package(OpenCV REQUIRED)

add_executable(main code/src/main.cpp
                    code/src/Utils.cpp
                    code/src/CardType.cpp
                    code/src/Loaders.cpp
                    code/src/Dataset.cpp)

target_include_directories(main PRIVATE code/include)
target_link_libraries(main ${OpenCV_LIBS})

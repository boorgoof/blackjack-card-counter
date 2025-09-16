cmake_minimum_required(VERSION 3.10)
project(EasyPeasyCV-Blackjack)

find_package(OpenCV REQUIRED)

add_executable(opencv_hello src/main.cpp)
target_link_libraries(opencv_hello ${OpenCV_LIBS})

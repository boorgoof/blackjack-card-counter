cmake_minimum_required(VERSION 3.10)
project(EasyPeasyCV-Blackjack)

find_package(OpenCV REQUIRED)

add_executable(main code/src/main.cpp
                    code/src/Utils.cpp
                    code/src/CardType.cpp
                    code/src/Loaders.cpp
                    code/src/Dataset.cpp)

add_executable(stats_test code/test/testStatsCalculation.cpp
                    code/src/Utils.cpp
                    code/src/CardType.cpp
                    code/src/Loaders.cpp
                    code/src/StatisticsCalculation.cpp)

target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(stats_test ${OpenCV_LIBS})

cmake_minimum_required(VERSION 3.10)
project(EasyPeasyCV-Blackjack)

find_package(OpenCV REQUIRED)

add_executable(main code/src/main.cpp
                    code/src/Utils.cpp
                    code/src/CardType.cpp
                    code/src/Loaders.cpp
                    code/src/Dataset.cpp
                    code/src/StatisticsCalculation.cpp
                    code/src/card_detector/SingleCardDetector.cpp
                    code/src/card_detector/SequentialCardDetector.cpp
                    code/src/card_detector/RoughCardDetector.cpp
                    code/src/card_detector/CardDetector.cpp
                    code/src/ImageFilter.cpp
                    )

add_executable(testGian code/src/mainGian.cpp
                    code/src/Utils.cpp
                    code/src/CardType.cpp
                    code/src/Loaders.cpp
                    code/src/Dataset.cpp
                    code/src/StatisticsCalculation.cpp
                    code/src/card_detector/SingleCardDetector.cpp
                    code/src/card_detector/SequentialCardDetector.cpp
                    code/src/card_detector/RoughCardDetector.cpp
                    code/src/card_detector/CardDetector.cpp
                    code/src/ImageFilter.cpp
                    )

add_executable(stats_test code/test/testStatsCalculation.cpp
                    code/src/Utils.cpp
                    code/src/CardType.cpp
                    code/src/Loaders.cpp
                    code/src/StatisticsCalculation.cpp)


set_property(TARGET main PROPERTY CXX_STANDARD 17)
set_property(TARGET testGian PROPERTY CXX_STANDARD 17)
set_property(TARGET stats_test PROPERTY CXX_STANDARD 17)

target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(stats_test ${OpenCV_LIBS})
target_link_libraries(testGian ${OpenCV_LIBS})

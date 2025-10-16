cmake_minimum_required(VERSION 3.10)
project(EasyPeasyCV-Blackjack)

find_package(OpenCV REQUIRED)



add_executable(testGian code/src/mainGian.cpp
                    code/src/Utils.cpp
                    code/src/CardType.cpp
                    code/src/Loaders.cpp
                    code/src/Dataset/Dataset.cpp
                    code/src/Dataset/ImageDataset.cpp
                    code/src/StatisticsCalculation.cpp
                    code/src/card_detector/SingleCardDetector.cpp
                    code/src/card_detector/SequentialCardDetector.cpp
                    code/src/card_detector/RoughCardDetector.cpp
                    code/src/card_detector/CardDetector.cpp
                    code/src/ImageFilter.cpp
                    )



set_property(TARGET testGian PROPERTY CXX_STANDARD 20)

target_link_libraries(testGian ${OpenCV_LIBS})

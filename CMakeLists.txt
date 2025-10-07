cmake_minimum_required(VERSION 3.10)
project(EasyPeasyCV-Blackjack)

find_package(OpenCV REQUIRED)

add_executable(main

    code/src/main.cpp
    code/src/Utils.cpp
    code/src/CardType.cpp
    code/src/Loaders.cpp
    code/src/Dataset.cpp
    code/src/StatisticsCalculation.cpp
    code/src/ImageFilter.cpp

    code/src/card_detector/CardDetector.cpp
    code/src/card_detector/SequentialCardDetector.cpp
    code/src/card_detector/SingleCardDetector.cpp

    code/src/card_detector/objectDetector/ObjectDetector.cpp
    code/src/card_detector/objectDetector/featurePipeline/FeatureExtractor.cpp
    code/src/card_detector/objectDetector/featurePipeline/FeatureMatcher.cpp
    code/src/card_detector/objectDetector/featurePipeline/FeaturePipeline.cpp
    code/src/card_detector/objectDetector/featurePipeline/features/FeatureContainer.cpp)

set_property(TARGET main PROPERTY CXX_STANDARD 17)
target_link_libraries(main ${OpenCV_LIBS})


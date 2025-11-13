cmake_minimum_required(VERSION 3.10)
project(EasyPeasyCV-Blackjack)

find_package(OpenCV REQUIRED)

add_executable(main

    code/src/main.cpp
    code/src/Utils.cpp
    code/src/ObjectType.cpp
    code/src/CardType.cpp
    code/src/Loaders.cpp
    code/src/CardProjection.cpp
    code/src/Dataset/ImageDataset.cpp
    code/src/Dataset/VideoDataset.cpp
    code/src/Dataset/TemplateDataset.cpp
    code/src/StatisticsCalculation.cpp
    code/src/ImageFilter.cpp

    code/src/card_detector/CardDetector.cpp
    code/src/card_detector/SequentialCardDetector.cpp
    code/src/card_detector/SingleCardDetector.cpp
    code/src/card_detector/RoughCardDetector.cpp

    code/src/card_detector/objectClassifiers/ObjectClassifier.cpp
    code/src/card_detector/objectClassifiers/featurePipeline/FeatureExtractor.cpp
    code/src/card_detector/objectClassifiers/featurePipeline/FeatureMatcher.cpp
    code/src/card_detector/objectClassifiers/featurePipeline/FeaturePipeline.cpp
    code/src/card_detector/objectClassifiers/featurePipeline/features/FeatureContainer.cpp
    
    code/src/card_detector/objectSegmenters/ObjectSegmenter.cpp
    code/src/card_detector/objectSegmenters/SimpleContoursCardSegmenter.cpp
    code/src/card_detector/objectSegmenters/DistanceTransformCardSegmenter.cpp
    code/src/card_detector/objectSegmenters/MultipleCardSegmenter.cpp)

set_property(TARGET main PROPERTY CXX_STANDARD 20)
target_link_libraries(main ${OpenCV_LIBS})





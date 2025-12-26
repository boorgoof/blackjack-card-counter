cmake_minimum_required(VERSION 3.10)
project(EasyPeasyCV-Blackjack)


find_package(OpenCV REQUIRED)

add_executable(main
    code/src/main.cpp
    code/src/Utils.cpp
    code/src/ObjectType.cpp
    code/src/CardType.cpp
    code/src/Loaders.cpp
    code/src/VideoWriter.cpp
    code/src/Dataset/ImageDataset.cpp
    code/src/Dataset/VideoDataset.cpp
    code/src/Dataset/TemplateDataset.cpp
    code/src/StatisticsCalculation.cpp
    code/src/ImageFilter.cpp

    code/src/detection/ProcessingMode.cpp
    code/src/detection/SequentialFrameProcessing.cpp
    code/src/detection/SingleFrameProcessing.cpp

    code/src/detection/card_detector/MaskCardDetector.cpp
    code/src/detection/card_detector/CardProjection.cpp
    code/src/detection/card_detector/CardDetector.cpp
    code/src/detection/card_detector/SegmentationClassificationCardDetector.cpp
    code/src/detection/card_detector/YoloCardDetector.cpp

    code/src/detection/card_detector/objectClassifiers/ObjectClassifier.cpp
    code/src/detection/card_detector/objectClassifiers/featurePipeline/FeatureExtractor.cpp
    code/src/detection/card_detector/objectClassifiers/featurePipeline/FeatureMatcher.cpp
    code/src/detection/card_detector/objectClassifiers/featurePipeline/FeaturePipeline.cpp
    code/src/detection/card_detector/objectClassifiers/featurePipeline/features/FeatureContainer.cpp

    code/src/detection/card_detector/objectSegmenters/ObjectSegmenter.cpp
    code/src/detection/card_detector/objectSegmenters/SimpleContoursCardSegmenter.cpp
    code/src/detection/card_detector/objectSegmenters/DistanceTransformCardSegmenter.cpp)

add_executable(main_gian
    code/src/mainGian.cpp
    code/src/Utils.cpp
    code/src/ObjectType.cpp
    code/src/CardType.cpp
    code/src/Loaders.cpp
    code/src/VideoWriter.cpp
    code/src/Dataset/ImageDataset.cpp
    code/src/Dataset/VideoDataset.cpp
    code/src/Dataset/TemplateDataset.cpp
    code/src/StatisticsCalculation.cpp
    code/src/ImageFilter.cpp

    code/src/detection/ProcessingMode.cpp
    code/src/detection/SequentialFrameProcessing.cpp
    code/src/detection/SingleFrameProcessing.cpp

    code/src/detection/card_detector/MaskCardDetector.cpp
    code/src/detection/card_detector/CardProjection.cpp
    code/src/detection/card_detector/CardDetector.cpp
    code/src/detection/card_detector/SegmentationClassificationCardDetector.cpp
    code/src/detection/card_detector/YoloCardDetector.cpp

    code/src/detection/card_detector/objectClassifiers/ObjectClassifier.cpp
    code/src/detection/card_detector/objectClassifiers/featurePipeline/FeatureExtractor.cpp
    code/src/detection/card_detector/objectClassifiers/featurePipeline/FeatureMatcher.cpp
    code/src/detection/card_detector/objectClassifiers/featurePipeline/FeaturePipeline.cpp
    code/src/detection/card_detector/objectClassifiers/featurePipeline/features/FeatureContainer.cpp

    code/src/detection/card_detector/objectSegmenters/ObjectSegmenter.cpp
    code/src/detection/card_detector/objectSegmenters/SimpleContoursCardSegmenter.cpp
    code/src/detection/card_detector/objectSegmenters/DistanceTransformCardSegmenter.cpp)


target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/code/include)
    
target_include_directories(main_gian PRIVATE ${CMAKE_SOURCE_DIR}/code/include)

target_link_libraries(main 
    ${OpenCV_LIBS} )

target_link_libraries(main_gian 
    ${OpenCV_LIBS} )
    
set_property(TARGET main PROPERTY CXX_STANDARD 20)
set_property(TARGET main_gian PROPERTY CXX_STANDARD 20)




#ifndef YOLO_CARD_DETECTOR_H
#define YOLO_CARD_DETECTOR_H

#include "CardDetector.h"

class YoloCardDetector : public CardDetector {
public:


    YoloCardDetector(const std::string& modelPath, bool visualize);
    ~YoloCardDetector() override = default;

    std::vector<Label> detect_cards(const cv::Mat& image) override;
    /**
     * @brief mapCardIndex: maps the index of the detected class from the model to the one used in the logic program. 
     *  This is necessary only if the model was trained on a different dataset with a different class ordering than the ones used in the program.
     */
    int mapCardIndex(int inputIndex);

private:
    cv::dnn::Net net;

};

#endif // YOLO_CARD_DETECTOR_H
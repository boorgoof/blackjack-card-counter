#include "../../../include/detection/card_detector/YoloCardDetector.h"
#include "../../../include/ObjectType.h"
#include "../../../include/CardType.h"

#include <opencv2/dnn.hpp>

YoloCardDetector::YoloCardDetector(const std::string& modelPath, bool visualize) 
    : CardDetector(visualize) {
    net = cv::dnn::readNetFromONNX(modelPath);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

std::vector<Label> YoloCardDetector::detect_cards(const cv::Mat& image) {
    
    std::vector<Label> detections;
    
    const float input_size = 1280.0f;
    const float conf_threshold = 0.4f;
    const float nms_threshold = 0.50f;

    // preprocessing of the image
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0/255.0, cv::Size(input_size, input_size), cv::Scalar(), true, false);
    
    // forward pass
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // post-processing: we obtian a 2D matrix where each row corresponds to a detection and the columns correspond to : center_x, center_y, width, height, conf_class_0, ...., conf_class_51
    cv::Mat res = outputs[0];
    if (res.dims == 3) {
        res = cv::Mat(res.size[1], res.size[2], CV_32F, res.ptr<float>());
    }
    cv::transpose(res, res); 

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // scale factors to convert back to original image size
    float x_factor = image.cols / input_size;
    float y_factor = image.rows / input_size;

    for (int i = 0; i < res.rows; ++i) {
        cv::Mat row = res.row(i);

        // we take the class with max score 
        cv::Mat scores = row.colRange(4, 56);  
        cv::Point class_id_point;
        double score;
        cv::minMaxLoc(scores, 0, &score, 0, &class_id_point);

        if (score > conf_threshold) {
            float cx = row.at<float>(0);
            float cy = row.at<float>(1);
            float w = row.at<float>(2);
            float h = row.at<float>(3);

            int left = static_cast<int>((cx - 0.5 * w) * x_factor);
            int top = static_cast<int>((cy - 0.5 * h) * y_factor);
            int width = static_cast<int>(w * x_factor);
            int height = static_cast<int>(h * y_factor);

            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back((float)score);
            class_ids.push_back(class_id_point.x);
        }
    }

    // Non-Maximum Suppression 
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);

    // Labels construction 
    for (int idx : indices) {

        CardType card = Yolo_index_codec::yolo_index_to_card(this->mapCardIndex(class_ids[idx]));
        std::vector<cv::Rect> bboxes = { boxes[idx] };
        detections.emplace_back(card.clone(), bboxes, confidences[idx]);
    }

    return detections;
}

int YoloCardDetector::mapCardIndex(int inputIndex) {
    
    // a mapping from the model's class indices to the ones used in the logic program (in CardType). 
    static const int mapping[52] = {
        37, 5, 9, 13, 17, 21, 25, 29, 33, 1, 41, 49, 45, 
        38, 6, 10, 14, 18, 22, 26, 30, 34, 2, 42, 50, 46, 
        39, 7, 11, 15, 19, 23, 27, 31, 35, 3, 43, 51, 47, 
        36, 4, 8, 12, 16, 20, 24, 28, 32, 0, 40, 48, 44   
    };

    if (inputIndex < 0 || inputIndex >= 52) {
        return -1;
    }

    return mapping[inputIndex];
}
#include "../../../include/detection/card_detector/YoloCardDetector.h"
#include "../../../include/ObjectType.h"
#include "../../../include/CardType.h"

#include <opencv2/dnn.hpp>

YoloCardDetector::YoloCardDetector(const std::string& modelPath, bool detect_full_card, bool visualize) {
    net = cv::dnn::readNetFromONNX(modelPath);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

std::vector<Label> YoloCardDetector::detect_cards(const cv::Mat& image) {
    std::vector<Label> detections;
    
    const float input_size = 1280.0f;
    const float conf_threshold = 0.7f;
    const float nms_threshold = 0.50f;

    // preprocessing
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0/255.0, cv::Size(input_size, input_size), cv::Scalar(), true, false);
    
    // forward pass
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // post-processing 
    cv::Mat res = outputs[0];
    if (res.dims == 3) {
        res = cv::Mat(res.size[1], res.size[2], CV_32F, res.ptr<float>());
    }
    cv::transpose(res, res);

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // Fattori di scala per riportare le coordinate alla dimensione dell'immagine originale
    float x_factor = image.cols / input_size;
    float y_factor = image.rows / input_size;

    for (int i = 0; i < res.rows; ++i) {
        cv::Mat row = res.row(i);
        // Le colonne 4-55 sono i punteggi delle 52 classi
        cv::Mat scores = row.colRange(4, 56);
        cv::Point class_id_point;
        double score;
        cv::minMaxLoc(scores, 0, &score, 0, &class_id_point);

        if (score > conf_threshold) {
            float cx = row.at<float>(0);
            float cy = row.at<float>(1);
            float w = row.at<float>(2);
            float h = row.at<float>(3);

            // Convertiamo da centro (cx, cy) a angolo (x, y) e scaliamo
            int left = static_cast<int>((cx - 0.5 * w) * x_factor);
            int top = static_cast<int>((cy - 0.5 * h) * y_factor);
            int width = static_cast<int>(w * x_factor);
            int height = static_cast<int>(h * y_factor);

            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back((float)score);
            class_ids.push_back(class_id_point.x);
        }
    }

    // Non-Maximum Suppression (NMS)
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);

    for (int idx : indices) {

        CardType card = Yolo_index_codec::yolo_index_to_card(this->mapCardIndex(class_ids[idx]));
        std::vector<cv::Rect> bboxes = { boxes[idx] };
        // Costruttore: Label(std::unique_ptr<ObjectType> obj, const std::vector<cv::Rect>& bboxes, float conf)
        detections.emplace_back(card.clone(), bboxes, confidences[idx]);
    }

    return detections;
}

int YoloCardDetector::mapCardIndex(int inputIndex) {
    
    static const int mapping[52] = {
        37, 5, 9, 13, 17, 21, 25, 29, 33, 1, 41, 49, 45, // C: 10, 2..9, A, J, K, Q
        38, 6, 10, 14, 18, 22, 26, 30, 34, 2, 42, 50, 46, // D: 10, 2..9, A, J, K, Q
        39, 7, 11, 15, 19, 23, 27, 31, 35, 3, 43, 51, 47, // H: 10, 2..9, A, J, K, Q
        36, 4, 8, 12, 16, 20, 24, 28, 32, 0, 40, 48, 44   // S: 10, 2..9, A, J, K, Q
    };

    if (inputIndex < 0 || inputIndex >= 52) {
        return -1;
    }

    return mapping[inputIndex];
}
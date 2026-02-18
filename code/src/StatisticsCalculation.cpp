#include "../include/StatisticsCalculation.h"
#include "../include/Label.h"
#include "../include/CardType.h"
#include <algorithm>
#include <unordered_set>
#include <map>
#include <vector>

float StatisticsCalculation::calc_IoU(const Label& true_label, const Label& pred_label) {
    
    const std::vector<cv::Rect>& true_boxes = true_label.get_bounding_boxes();
    const std::vector<cv::Rect>& pred_boxes = pred_label.get_bounding_boxes();

     if (true_boxes.size() != 1 || pred_boxes.empty())
        return 0.0f;
    
    cv::Rect2f trueRect(true_boxes[0]);
    float best_iou = 0.0f;

    for (const cv::Rect& pred_box : pred_boxes) {
        
        cv::Rect2f predRect(pred_box);

        cv::Rect2f intersection = trueRect & predRect;
        float intersection_area = std::max(0.0f, intersection.area());

        float union_area = std::max(0.0f, trueRect.area() + predRect.area() - intersection_area);
        if (union_area == 0.0f)
            continue;

        float iou = intersection_area / union_area;
        if (iou > best_iou)
            best_iou = iou;
    }

    return best_iou;
}

std::vector<float> StatisticsCalculation::calc_image_meanIoU(const std::vector<Label>& true_labels, const std::vector<Label>& predicted_labels)
{
    
    // Case with no objects in both true and predicted labels. We define mean IoU = 1.0
    if (true_labels.empty() && predicted_labels.empty()) {
        return {}; 
    }

    // Case with no objects in just one of true and predicted labels. We define mean IoU = 0.0
    if (true_labels.empty() || predicted_labels.empty()) {
        size_t n = std::max(true_labels.size(), predicted_labels.size());
        return std::vector<float>(n, 0.0f); 
    }

    // 1) calculate IoU for each pair of true and predicted labels
    std::vector<std::tuple<float,int,int>> all_candidates_pairs; 
    all_candidates_pairs.reserve(true_labels.size() * predicted_labels.size());

    for (int gt_idx = 0; gt_idx < static_cast<int>(true_labels.size()); ++gt_idx) {
        for (int pred_idx = 0; pred_idx < static_cast<int>(predicted_labels.size()); ++pred_idx) {
            
            float IoU = StatisticsCalculation::calc_IoU(true_labels[gt_idx], predicted_labels[pred_idx]);
            if (IoU > 0.0f) {
                all_candidates_pairs.emplace_back(IoU, gt_idx, pred_idx);
            }
        }
    }

    if (all_candidates_pairs.empty()) {
        size_t n = std::max(true_labels.size(), predicted_labels.size());
        return std::vector<float>(n, 0.0f); 
    }

    //2) Sort candidates by IoU in descending order
    std::sort(all_candidates_pairs.begin(), all_candidates_pairs.end(), [](const auto& trueRect, const auto& predRect){ return std::get<0>(trueRect) > std::get<0>(predRect); });

    //3) Greedy assignment for matching true and predicted labels: one true label is matched with the predicted label with the highest IoU
    std::vector<char> true_used(true_labels.size(), 0);
    std::vector<char> pred_used(predicted_labels.size(), 0);

    size_t n = std::max(true_labels.size(), predicted_labels.size());
    std::vector<float> objects_iou(n, 0.0f);
    //int predictions = 0;

    for (const auto& candidate : all_candidates_pairs) {

        int true_idx = std::get<1>(candidate);
        int pred_idx = std::get<2>(candidate);
        if (true_used[true_idx] || pred_used[pred_idx]) continue; // already used labels in trueRect match

        float IoU = std::get<0>(candidate);
        objects_iou[pred_idx] = IoU;

        true_used[true_idx] = pred_used[pred_idx] = 1;

    }

    return objects_iou;
}


// usefull link: https://medium.com/mcd-unison/multiclass-confusion-matrix-clarity-without-confusion-88af1494c1d1
cv::Mat StatisticsCalculation::calc_confusion_matrix(const std::vector<Label>& true_labels,
                                                    const std::vector<Label>& pred_labels,
                                                    int num_classes,
                                                    float iou_threshold)
{
    cv::Mat multiclass_conf_matrix = cv::Mat::zeros(num_classes, num_classes, CV_32S); // rows: predicted class, cols: actual class
    if (num_classes <= 0) return multiclass_conf_matrix;

    const int no_object_index = num_classes - 1; // last row/column: case where there is no object in the image

    // 1) calculate IoU for each pair of true and predicted labels
    std::vector<std::tuple<float,int,int>> all_candidates_pairs;
    all_candidates_pairs.reserve(true_labels.size() * pred_labels.size());

    for (int i = 0; i < static_cast<int>(true_labels.size()); ++i) {
        for (int j = 0; j < static_cast<int>(pred_labels.size()); ++j) {
            float IoU = StatisticsCalculation::calc_IoU(true_labels[i], pred_labels[j]);
            if (IoU >= iou_threshold) {
                 all_candidates_pairs.emplace_back(IoU, i, j);
            }
        }
    }

    //2) Sort candidates by IoU in descending order
    std::sort(all_candidates_pairs.begin(), all_candidates_pairs.end(), [](const auto& trueRect, const auto& predRect) { return std::get<0>(trueRect) > std::get<0>(predRect); });

    //3) Greedy assignment for matching true and predicted labels: one true label is matched with the predicted label with the highest IoU
    //   We handle the True Positive case and Mispredicted case (FP) 
    std::vector<bool> true_used(true_labels.size(), false);
    std::vector<bool> pred_used(pred_labels.size(), false);
    
    for (const auto& candidate : all_candidates_pairs) {

        int true_idx = std::get<1>(candidate);
        int pred_idx = std::get<2>(candidate);
        if (true_used[true_idx] || pred_used[pred_idx]) continue; // already used labels in trueRect match
                
        int row_predicted_class_index = pred_labels[pred_idx].get_object()->get_id_number();
        int col_actual_class_index = true_labels[true_idx].get_object()->get_id_number();

        //CV_Assert(0 <= row_predicted_class_index && row_predicted_class_index < num_classes);
        //CV_Assert(0 <= col_actual_class_index && col_actual_class_index < num_classes);
        
        multiclass_conf_matrix.at<int>(row_predicted_class_index, col_actual_class_index) += 1;

        true_used[true_idx] = 1;
        pred_used[pred_idx] = 1;
    }


    // 4) Undetected object (FN): we have trueRect true label that does not have trueRect corresponding predicted label: row = no_object_index, col = col_actual_class_index
    for (int true_idx = 0; true_idx < (int)true_labels.size(); ++true_idx) {
       
        if (!true_used[true_idx]) {
            
            int col_actual_class_index = true_labels[true_idx].get_object()->get_id_number();
            
            //CV_Assert(0 <= col_actual_class_index && col_actual_class_index < num_classes);
            
            multiclass_conf_matrix.at<int>(no_object_index, col_actual_class_index) += 1; 
        }
    }

    // 5) Ghost prediction (FP but sightly different form Mispredicted case): row = row_predicted_class_index, col = no_object_index
    for (int pred_idx = 0; pred_idx < (int)pred_labels.size(); ++pred_idx) {
        if (!pred_used[pred_idx]) {

            int row_predicted_class_index = pred_labels[pred_idx].get_object()->get_id_number();
            
            CV_Assert(0 <= row_predicted_class_index && row_predicted_class_index < num_classes);
            
            multiclass_conf_matrix.at<int>(row_predicted_class_index, no_object_index) += 1; // (pred class, no annotation)
        }
    }

    return multiclass_conf_matrix;

}

cv::Mat StatisticsCalculation::calc_confusion_matrix(const std::vector<std::vector<Label>>& true_labels_dataset,
                                const std::vector<std::vector<Label>>& pred_labels_dataset,
                                int num_classes,
                                float iou_threshold) {

    if (true_labels_dataset.size() != pred_labels_dataset.size()) {
        throw std::invalid_argument("calc_confusion_matrix: the two input vectors must have the same dimension.");
    }

    // multiclass confusion matrix
    cv::Mat mcm = cv::Mat::zeros(num_classes, num_classes, CV_32S);
    for (size_t i = 0; i < true_labels_dataset.size(); ++i) {
        mcm += StatisticsCalculation::calc_confusion_matrix(true_labels_dataset[i], pred_labels_dataset[i], num_classes, iou_threshold);
    }
    return mcm;
}

float StatisticsCalculation::calc_accuracy(const cv::Mat &confusion_matrix)
{
    CV_Assert(confusion_matrix.rows == confusion_matrix.cols);
    CV_Assert(confusion_matrix.type() == CV_32S);

    int TP = 0;
    int TN = 0;
    int FP = 0;
    int FN = 0;

    for (int i = 0; i < confusion_matrix.rows; ++i) {
        for (int j = 0; j < confusion_matrix.cols; ++j) {
            if (i == j) {
                TP += confusion_matrix.at<int>(i, j);
            } else {
                FP += confusion_matrix.at<int>(i, j);
                FN += confusion_matrix.at<int>(j, i);
            }
        }
    }

    return static_cast<float>(TP + TN) / (TP + TN + FP + FN);
}

// usefull link: https://medium.com/mcd-unison/multiclass-confusion-matrix-clarity-without-confusion-88af1494c1d1
std::vector<float> StatisticsCalculation::calc_precision(const cv::Mat& confusion_matrix){
    
    CV_Assert(confusion_matrix.rows == confusion_matrix.cols);
    CV_Assert(confusion_matrix.type() == CV_32S);

    int matrix_dim = confusion_matrix.rows;
    int label_classes = confusion_matrix.rows - 1 ;

    std::vector<float> precision(label_classes, 0.0f); // precision = TP / (TP + FP)


    for (int c = 0; c < label_classes; ++c) {

        long long true_positive = confusion_matrix.at<int>(c, c);

        long long row_sum = 0; // all predicted as class c
        for (int j = 0; j < matrix_dim; ++j) {
            row_sum += confusion_matrix.at<int>(c, j);
        }

        //const long long false_positive = row_sum - true_positive;
        const long long denom = row_sum;
        precision[c] = (denom > 0) ? static_cast<float>(static_cast<double>(true_positive) / static_cast<double>(denom)) : 0.0f;
    }

    return precision;
}

std::vector<float> StatisticsCalculation::calc_recall(const cv::Mat& confusion_matrix)
{
    CV_Assert(confusion_matrix.rows == confusion_matrix.cols);
    CV_Assert(confusion_matrix.type() == CV_32S);

    int matrix_dim = confusion_matrix.rows;
    int label_classes = confusion_matrix.rows - 1 ;

    std::vector<float> recall(label_classes, 0.0f);

    for (int c = 0; c < label_classes; ++c) {

        long long true_positive = confusion_matrix.at<int>(c, c);

        long long col_sum = 0; // all actual class c
        for (int i = 0; i < matrix_dim; ++i) {
            col_sum += confusion_matrix.at<int>(i, c);
        }

        //const long long false_negative = col_sum - true_positive;
        const long long denom = col_sum;
        recall[c] = (denom > 0) ? static_cast<float>(static_cast<double>(true_positive) / static_cast<double>(denom)) : 0.0f;
    }

    return recall;
}

std::vector<float> StatisticsCalculation::calc_f1(const cv::Mat& confusion_matrix)
{
    int matrix_dim    = confusion_matrix.rows;
    int num_label_classes = std::max(0, matrix_dim - 1);

    std::vector<float> precisions = StatisticsCalculation::calc_precision(confusion_matrix);
    std::vector<float> recalls = StatisticsCalculation::calc_recall(confusion_matrix);

    std::vector<float> f1_scores(num_label_classes, 0.0f);

    for (int c = 0; c < num_label_classes; ++c) {

        double p = static_cast<double>(precisions[c]);
        double r = static_cast<double>(recalls[c]);
        double denom = p + r;

        f1_scores[c] = (denom > 0.0) ? static_cast<float>(2.0 * p * r / denom) : 0.0f;
    }
    return f1_scores;
}


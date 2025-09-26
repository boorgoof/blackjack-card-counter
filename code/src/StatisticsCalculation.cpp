#include "../include/StatisticsCalculation.h"
#include "../include/Label.h"
#include <algorithm>
#include <unordered_set>
#include <map>
#include <vector>

float StatisticsCalculation::calc_IoU(const Label& true_label, const Label& pred_label) {
    
    // Insection area
    cv::Rect intersectionRect = pred_label.get_bounding_box() & true_label.get_bounding_box();
    double intersection_area = intersectionRect.area();

    // Union area
    double union_area = true_label.get_bounding_box().area() + true_label.get_bounding_box().area() - intersection_area;

    // IoU
    return intersection_area / union_area;
}

float StatisticsCalculation::calc_image_meanIoU(const std::vector<Label>& true_labels,const std::vector<Label>& predicted_labels)
{
    
    // Case with no objects in both true and predicted labels. We define mean IoU = 1.0
    if (true_labels.empty() && predicted_labels.empty()) {
        return 1.0f; 
    }

    // Case with no objects in just one of true and predicted labels. We define mean IoU = 0.0
    if (true_labels.empty() || predicted_labels.empty()) {
        return 0.0f; 
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
        return 0.0f; 
    }

    //2) Sort candidates by IoU in descending order
    std::sort(all_candidates_pairs.begin(), all_candidates_pairs.end(), [](const auto& a, const auto& b){ return std::get<0>(a) > std::get<0>(b); });

    //3) Greedy assignment for matching true and predicted labels: one true label is matched with the predicted label with the highest IoU
    std::vector<char> true_used(true_labels.size(), 0);
    std::vector<char> pred_used(predicted_labels.size(), 0);

    double sum_iou = 0.0;
    //int predictions = 0;

    for (const auto& candidate : all_candidates_pairs) {

        int true_idx = std::get<1>(candidate);
        int pred_idx = std::get<2>(candidate);
        if (true_used[true_idx] || pred_used[pred_idx]) continue; // already used labels in a match

        float IoU = std::get<0>(candidate);
        sum_iou += static_cast<double>(IoU);
        //predictions += 1;

        true_used[true_idx] = pred_used[pred_idx] = 1;

    }

    int predictions = std::max(true_labels.size(), predicted_labels.size());
    return (predictions > 0) ? static_cast<float>(sum_iou / predictions) : 0.0f;

}

std::vector<float> StatisticsCalculation::calc_dataset_meanIoU(const std::vector<std::vector<Label>>& true_labels_per_image,
                                                                const std::vector<std::vector<Label>>& predicted_labels_per_image)
{
    if (true_labels_per_image.size() != predicted_labels_per_image.size()) {
        throw std::invalid_argument("True and predicted label lists must have the same number of images.");
    }

    std::vector<float> mean_IoU_list;
    mean_IoU_list.reserve(true_labels_per_image.size());

    for (size_t img_idx = 0; img_idx < true_labels_per_image.size(); ++img_idx) {

        float image_mean_iou = StatisticsCalculation::calc_image_meanIoU(true_labels_per_image[img_idx], predicted_labels_per_image[img_idx] );
        mean_IoU_list.push_back(image_mean_iou);
    }

    return mean_IoU_list;
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
    std::sort(all_candidates_pairs.begin(), all_candidates_pairs.end(), [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });

    //3) Greedy assignment for matching true and predicted labels: one true label is matched with the predicted label with the highest IoU
    //   We handle the True Positive case and Mispredicted case (FP) 
    std::vector<bool> true_used(true_labels.size(), false);
    std::vector<bool> pred_used(pred_labels.size(), false);
    
    for (const auto& candidate : all_candidates_pairs) {

        int true_idx = std::get<1>(candidate);
        int pred_idx = std::get<2>(candidate);
        if (true_used[true_idx] || pred_used[pred_idx]) continue; // already used labels in a match
                                                                  
        int col_actual_class_index = Yolo_index_codec::card_to_yolo_index(true_labels[true_idx].get_class_name());
        int row_predicted_class_index = Yolo_index_codec::card_to_yolo_index(pred_labels[pred_idx].get_class_name());

        //CV_Assert(0 <= row_predicted_class_index && row_predicted_class_index < num_classes);
        //CV_Assert(0 <= col_actual_class_index && col_actual_class_index < num_classes);
        
        multiclass_conf_matrix.at<int>(col_actual_class_index, row_predicted_class_index) += 1;

        true_used[true_idx] = 1;
        pred_used[pred_idx] = 1;
    }


    // 4) Undetected object (FN): we have a true label that does not have a corresponding predicted label: row = no_object_index, col = col_actual_class_index
    for (int true_idx = 0; true_idx < (int)true_labels.size(); ++true_idx) {
       
        if (!true_used[true_idx]) {
            
            int col_actual_class_index = Yolo_index_codec::card_to_yolo_index(true_labels[true_idx].get_class_name());
            
            //CV_Assert(0 <= col_actual_class_index && col_actual_class_index < num_classes);
            
            multiclass_conf_matrix.at<int>(no_object_index, col_actual_class_index) += 1; 
        }
    }

    // 5) Ghost prediction (FP but sightly different form Mispredicted case): riga = row_predicted_class_index, col = no_object_index
    for (int pred_idx = 0; pred_idx < (int)pred_labels.size(); ++pred_idx) {
        if (!pred_used[pred_idx]) {

            int row_predicted_class_index = Yolo_index_codec::card_to_yolo_index(pred_labels[pred_idx].get_class_name());
            
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

        const long long false_positive = row_sum - true_positive;
        const long long denom = true_positive + false_positive;
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

        long long false_negative = col_sum - true_positive;
        long long denom = true_positive + false_negative;
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


void StatisticsCalculation::print_confusion_matrix(const cv::Mat& confusion_matrix)
{
    CV_Assert(confusion_matrix.rows == confusion_matrix.cols);
    CV_Assert(confusion_matrix.type() == CV_32S);

    int matrix_dim = confusion_matrix.rows;

    std::cout << "Confusion Matrix (" << matrix_dim << " x " << matrix_dim << ") " << std::endl;
    std::cout <<  "[rows = predicted, cols = ground truth]" << std::endl;

    // Columns indices
    int weight_cell = 8;
    std::cout << std::setw(weight_cell) << "";
    for (int j = 0; j < matrix_dim; ++j) std::cout << std::setw(weight_cell) <<  Yolo_index_codec::yolo_index_to_card(j);
    std::cout << '\n';

    // Rows
    for (int i = 0; i < matrix_dim; ++i) {

        std::cout << std::setw(weight_cell) << Yolo_index_codec::yolo_index_to_card(i);

        for (int j = 0; j < matrix_dim; ++j) {
            std::cout << std::setw(weight_cell) << confusion_matrix.at<int>(i, j);
        }
        std::cout << '\n';
    }
}

void StatisticsCalculation::print_metrics(const std::vector<float>& precision,
                                          const std::vector<float>& recall,
                                          const std::vector<float>& f1_scores)
{

    CV_Assert(precision.size() == recall.size());
    CV_Assert(precision.size() == f1_scores.size());

    const size_t num_label_classes = precision.size();
    if (num_label_classes == 0) {
        std::cout << "No metrics to print: 0 classes." << std::endl;
        return;
    }

    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(4);

    const int col_width_class = 8;
    const int col_width_value = 12;


    // header
    std::cout << std::setw(col_width_class) << "Class"
              << std::setw(col_width_value) << "Precision"
              << std::setw(col_width_value) << "Recall"
              << std::setw(col_width_value) << "F1" << '\n';

    
    double sum_precision = 0.0, sum_recall = 0.0, sum_f1 = 0.0;

    // metrics values
    for (size_t class_index = 0; class_index < num_label_classes; ++class_index) {
        std::cout << std::setw(col_width_class) << class_index
                  << std::setw(col_width_value) << precision[class_index]
                  << std::setw(col_width_value) << recall[class_index]
                  << std::setw(col_width_value) << f1_scores[class_index]
                  << '\n';

        sum_precision += precision[class_index];
        sum_recall    += recall[class_index];
        sum_f1        += f1_scores[class_index];
    }

    // mean meatrics values
    std::cout << std::string(col_width_class + 3 * col_width_value, '-') << '\n';
    std::cout << std::setw(col_width_class) << "Mean"
              << std::setw(col_width_value) << static_cast<float>(sum_precision / num_label_classes)
              << std::setw(col_width_value) << static_cast<float>(sum_recall / num_label_classes)
              << std::setw(col_width_value) << static_cast<float>(sum_f1 / num_label_classes)
              << '\n';

}
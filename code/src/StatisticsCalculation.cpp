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


std::vector<float> StatisticsCalculation::calc_dataset_meanIoU(const std::vector<std::vector<Label>>& true_labels,
                                                                const std::vector<std::vector<Label>>& pred_labels) {
    if (true_labels.size() != pred_labels.size()) {
        throw std::invalid_argument("True and predicted labels vectors must have the same size");
    }
    
    std::vector<float> IoU_list;
    IoU_list.reserve(true_labels.size());
    
    // calculate mean IoU for each image
    for (size_t img_idx = 0; img_idx < true_labels.size(); ++img_idx) {

        float image_iou = calc_image_meanIoU(true_labels[img_idx], pred_labels[img_idx]);
        IoU_list.push_back(image_iou);

    }
    
    return IoU_list;
}

// In one image, we can have multiple objects and some of them can be of the same class
// What we have is a list of true labels and a list of predicted labels, and the goal to calculate the mean IoU for all objects in the image
// But when we calculate IoU, we need to find for each predicted label the corresponding true label that it is predicting.
// We cannot calculate IoU using one predicted label with a true labal which is not the one it is predicting.
// What we do is to group the labels by class. Then
// If there is just one predicted label in a class, we can directly calculate IoU
// If there are multiple predicted labels in a class, we need to find the best matches between true and predicted labels
float StatisticsCalculation::calc_image_meanIoU(const std::vector<Label>& true_labels,
                                                const std::vector<Label>& pred_labels) {

    // If we have no objects in both, we consider it as a perfect match
    if (true_labels.empty() && pred_labels.empty()) {
        return 1.0f;
    }
    
    // If one group of labels is empty, we consider no match
    if (true_labels.empty() || pred_labels.empty()) {
        return 0.0f;
    }
    
    // If there are objects, initially we group them by class
    std::map<Card_Type, std::vector<Label>> true_labels_per_class = StatisticsCalculation::Helper::group_labels_by_class(true_labels);
    std::map<Card_Type, std::vector<Label>> pred_labels_per_class = StatisticsCalculation::Helper::group_labels_by_class(pred_labels);
    
    float tot_IoU = 0.0f;
    int tot_predictions = 0;
    
    // we process each class separately
    for (const auto& [card_type, true_class_labels] : true_labels_per_class) {
        
        auto it = pred_labels_per_class.find(card_type);

        // if there are no predicted labels for this class, skip
        if (it == pred_labels_per_class.end()) {
            continue;
        }
        
        // If there are predicted labels for this class, we find the best matches between true and predicted labels.
        // Note the trivial case of just one predicted label in a class is also handled in the function: If there is just one predicted label in a class, we can directly calculate IoU
        const std::vector<Label>& pred_class_labels = it->second;
        std::vector<std::pair<int,int>> matches = StatisticsCalculation::Helper::find_best_labels_pairs(true_class_labels, pred_class_labels);
        
        // we sum IoU for each matched pair: true and predicted label 
        for (const auto& [true_idx, pred_idx] : matches) {
            float IoU = calc_IoU(true_class_labels[true_idx], pred_class_labels[pred_idx]);
            tot_IoU += IoU;
            tot_predictions++;
        }
    }
    
    return tot_predictions > 0 ? tot_IoU / tot_predictions : 0.0f;
}

// Group labels by their class (Card_Type)
std::map<Card_Type, std::vector<Label>> StatisticsCalculation::Helper::group_labels_by_class(const std::vector<Label>& labels) {
    
    std::map<Card_Type, std::vector<Label>> labels_per_class;
    
    for (const auto& label : labels) {
        labels_per_class[label.get_class_name()].push_back(label);
    }
    
    return labels_per_class;
}


// One image can have multiple objects in the same class. 
// So we need to find for each predicted label the corrisponding true label.
// It return the list of index pairs (true_label_index, pred_label_index) 
std::vector<std::pair<int,int>> StatisticsCalculation::Helper::find_best_labels_pairs(const std::vector<Label>& true_labels,
                                                        const std::vector<Label>& pred_labels) {

    std::vector<std::pair<int,int>> matches;
    if (true_labels.empty() || pred_labels.empty()) {
        return matches;
    }
    
    // Trivial case: just one predicted label and one true label
    if (true_labels.size() == 1 && pred_labels.size() == 1) {
        matches.emplace_back(0, 0);
        return matches;
    }
    
    // General case: multiple true and predicted labels in the same class
    // 1) calculate IoU for each pair of true and predicted labels
    std::vector<std::tuple<float, int, int>> all_candidates_pairs;
    for (size_t i = 0; i < true_labels.size(); ++i) {
        for (size_t j = 0; j < pred_labels.size(); ++j) {
            float iou = StatisticsCalculation::calc_IoU(true_labels[i], pred_labels[j]);
            all_candidates_pairs.emplace_back(iou, i, j);
        }
    }
    
    //2) Sort candidates by IoU in descending order
    std::sort(all_candidates_pairs.begin(), all_candidates_pairs.end(), 
              [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
    
    //3) Greedy assignment
    std::vector<bool> true_used(true_labels.size(), false);
    std::vector<bool> pred_used(pred_labels.size(), false);
    
    for (const auto& [iou, true_idx, pred_idx] : all_candidates_pairs) {
        
        if (!true_used[true_idx] && !pred_used[pred_idx]) {
            matches.emplace_back(true_idx, pred_idx);
            true_used[true_idx] = true;
            pred_used[pred_idx] = true;
        }
    }
    
    return matches;
}


/*

cv::Mat StatisticsCalculation::calc_confusion_matrix(const std::vector<std::vector<Label>>& true_labels,
                                                    const std::vector<std::vector<Label>>& pred_labels,
                                                    int num_classes) {
        
    if (true_labels.size() != pred_labels.size()) {
        throw std::invalid_argument("True and predicted labels vectors must have the same size");
    }
    
    cv::Mat confusion_matrix = cv::Mat::zeros(num_classes, num_classes, CV_32S);
    
    // Process each image
    for (size_t img_idx = 0; img_idx < true_labels.size(); ++img_idx) {
        cv::Mat single_image_matrix = calc_confusion_matrix(
            true_labels[img_idx], 
            pred_labels[img_idx], 
            num_classes, 
            0.5f  
        );
        
        // Accumulate into total matrix
        confusion_matrix += single_image_matrix;
    }
    
    return confusion_matrix;
}*/
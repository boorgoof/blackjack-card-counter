#ifndef STATISTICS_CALCULATION_H
#define STATISTICS_CALCULATION_H

#include "../include/Label.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <iomanip>
#include <stdexcept>

/**
 * @brief Namespace to calculate evaluation metrics such as 
 *  IoU, precision, recall, F1-score and confusion matrices fot detection tasks.
 */
namespace StatisticsCalculation {
    
    
    /**
     * @brief Calculates IoU (Intersection over Union) between the bounding boxes of two labels.
     * @param true_label the real label (ground truth)
     * @param  pred_label the predicted label
     * @return the IoU value 
     */
    float calc_IoU(const Label& true_label, const Label& pred_label);


    /**
     * @brief Calculates the mean IoU for one image.
     * First calcultes IoU for each pair of true and predicted labels in a image and then computes the mean. 
     * 
     * @param true_masks vector of ground truth masks
     * @param pred_masks vector of the predicted labels
     * @return float mean IoU value of all objects in the image
     * @throws std::invalid_argument If vectors have different sizes
     */                                   
    float calc_image_meanIoU(const std::vector<Label>& true_labels,
                                                        const std::vector<Label>& pred_labels);
    
    /**
     * @brief Calculates IoU list for each image of the dataset.
     * It computes calc_image_meanIoU for each image of the dataset.
     * 
     * @param true_masks vector of ground truth masks
     * @param pred_masks vector of the predicted labels
     * @return std::vector<float> List of IoU values
     * @throws std::invalid_argument If vectors have different sizes
     */
    std::vector<float> calc_dataset_meanIoU(const std::vector<std::vector<Label>>& true_labels, 
                                       const std::vector<std::vector<Label>>& pred_labels);
    
    //TODO: instead of asking the number of classes, ask for the list of classes (std::vector<Card_Type>) to be more flexible
     /**
     * @brief Calculates confusion matrix (TP, FP, FN, TN) from single image labels
     * 
     * @param true_labels vector of ground truth labels
     * @param pred_labels vector of predicted labels
     * @param num_classes number of classes
     * @param iou_threshold IoU threshold to consider a pred_label as positive
     * @return cv::Mat confusion matrix (num_classes x num_classes). Note: (actual x predicted)
     */
    cv::Mat calc_confusion_matrix(const std::vector<Label>& true_labels, 
                                 const std::vector<Label>& pred_labels, 
                                 int num_classes,
                                 float iou_threshold = 0.5f);
    
    //TODO: instead of asking the number of classes, ask for the list of classes (std::vector<Card_Type>) to be more flexible
    /**
     * @brief Calculates confusion matrix from multiple images
     *      
     * @param true_labels vector of ground truth labels for each image
     * @param pred_labels vector of predicted labels for each image
     * @param num_classes the number of classes
     * @param iou_threshold IoU threshold to consider a pred_label as positive
     * @return cv::Mat confusion matrix which consider all images (num_classes x num_classes).  Note: (actual x predicted)
     */
    cv::Mat calc_confusion_matrix(const std::vector<std::vector<Label>>& true_labels,
                                 const std::vector<std::vector<Label>>& pred_labels,
                                 int num_classes);
    
    /**
     * @brief Calculates precision for each class from confusion matrix.
     * Precision = TP / (TP + FP)
     * 
     * 
     * @param confusion_matrix the confusion matrix
     * @param label_classes number of classes
     * @return std::vector<float> of precisions: one for each class
     */
    std::vector<float> calc_precision(const cv::Mat& confusion_matrix, int label_classes);
    
    /**
     * @brief Calculates recall for each class from confusion matrix
     * Recall = TP / (TP + FN)
     * 
     * 
     * @param confusion_matrix confusion matrix
     * @param label_classes number of label classes
     * @return std::vector<float> of recalls: one for each class
     */
    std::vector<float> calc_recall(const cv::Mat& confusion_matrix, int label_classes);
    
    /**
     * @brief Calculates F1-score for each class from confusion matrix
     * F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
     * 
     * @param confusion_matrix confusion matrix
     * @param label_classes number of label classes
     * @return std::vector<float> of F1-scores for each class
     */
    std::vector<float> calc_f1(const cv::Mat& confusion_matrix, int label_classes);
    
    
    /**
     * @brief Utility function to print confusion matrix
     * 
     * @param confusion_matrix Confusion matrix to print
     */
    void print_confusion_matrix(const cv::Mat& confusion_matrix);
    
    /**
     * @brief Utility function to print metrics
     *
     * @param precision vector of precisions per class
     * @param recall vector of recalls per class
     * @param f1_scores vector of F1-scores per class
     */
    void print_metrics(const std::vector<float>& precision, 
                      const std::vector<float>& recall, 
                      const std::vector<float>& f1_scores);
    
    /**
     * @brief Internal helper functions 
     */
    namespace Helper {
        
        /**
         * @brief Groups labels by their Card_Type class
         * @param labels vector of labels that we want to group
         * @return map where key=Card_Type, value=vector of labels of that class
         */
        std::map<Card_Type, std::vector<Label>> group_labels_by_class(const std::vector<Label>& labels);
        
        /**
         * @brief Finds optimal matches between labels of the same class using greedy algorithm
         * @param true_labels vector of ground truth labels. Note: all labels must be of the same class
         * @param pred_labels vector of predicted labels. Note: all labels must be of the same class
         * @return vector of index pairs (true_idx, pred_idx) representing matches
         */
        std::vector<std::pair<int, int>> find_best_labels_pairs(const std::vector<Label>& true_labels,
                                                                    const std::vector<Label>& pred_labels);
        
        /**
         * @brief Converts Card_Type to class index for confusion matrix
         * @param card_type the card type to convert
         * @return int class index (0-51 for standard deck)
         */
        int card_type_to_class_index(const Card_Type& card_type);
        
        /**
         * @brief Converts class index back to Card_Type  
         * @param class_index the class index (0-51)
         * @return Card_Type corresponding card type
         */
        Card_Type class_index_to_card_type(int class_index);
        
    } 

} 

#endif // STATISTICS_CALCULATION_H
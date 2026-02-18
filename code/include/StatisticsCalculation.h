// Federico Meneghetti

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
     * @brief Calculates the mean IoU for all objects in one image.
     * For a true label it select the best match across all classes, not just within the same class
     * 
     * @param true_labels vector of ground truth labels
     * @param predicted_labels vector of the predicted labels
     * @return std::vector<float> mean IoU value of all objects in the image
     */                                   
    std::vector<float> calc_image_meanIoU(const std::vector<Label>& true_labels, const std::vector<Label>& predicted_labels);
    
    
     /**
     * @brief Calculates confusion matrix (TP, FP, FN, TN) from single image labels
     * 
     * @param true_labels vector of ground truth labels
     * @param pred_labels vector of predicted labels
     * @param num_classes number of classes
     * @param iou_threshold IoU threshold to consider a pred_label as positive
     * @return cv::Mat confusion matrix (num_classes x num_classes). Note: (actual x predicted)
     */
    cv::Mat calc_confusion_matrix(const std::vector<Label>& true_labels, const std::vector<Label>& pred_labels, int num_classes, float iou_threshold = 0.5f);
    
    /**
     * @brief Calculates confusion matrix from a dataset (multiple images)
     *      
     * @param true_labels vector of ground truth labels for each image
     * @param pred_labels vector of predicted labels for each image
     * @param num_classes the number of classes
     * @param iou_threshold IoU threshold to consider a pred_label as positive
     * @return cv::Mat confusion matrix which consider all images (num_classes x num_classes).  Note: (actual x predicted)
     */
    cv::Mat calc_confusion_matrix(const std::vector<std::vector<Label>>& true_labels, const std::vector<std::vector<Label>>& pred_labels, int num_classes = 53,  float iou_threshold = 0.5f);
    
    /**
     * @brief Calculates accuracy from confusion matrix
     * Accuracy = (TP + TN) / (TP + TN + FP + FN)
     * 
     * @param confusion_matrix the confusion matrix
     * @return float accuracy value
     */
    float calc_accuracy(const cv::Mat& confusion_matrix);
    /**
     * @brief Calculates precision for each class from confusion matrix.
     * Precision = TP / (TP + FP)
     * 
     * 
     * @param confusion_matrix the confusion matrix
     * @return std::vector<float> of precisions: one for each class
     */
    std::vector<float> calc_precision(const cv::Mat& confusion_matrix);
    
    /**
     * @brief Calculates recall for each class from confusion matrix
     * Recall = TP / (TP + FN)
     * 
     * 
     * @param confusion_matrix confusion matrix
     * @return std::vector<float> of recalls: one for each class
     */
    std::vector<float> calc_recall(const cv::Mat& confusion_matrix);
    
    /**
     * @brief Calculates F1-score for each class from confusion matrix
     * F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
     * 
     * @param confusion_matrix confusion matrix
     * @return std::vector<float> of F1-scores for each class
     */
    std::vector<float> calc_f1(const cv::Mat& confusion_matrix);
    
    
    
} 

#endif // STATISTICS_CALCULATION_H
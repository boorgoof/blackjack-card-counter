#ifndef CARDPROJECTION_H
#define CARDPROJECTION_H

#include <opencv2/opencv.hpp>

namespace CardProjection {

/**
 * @brief flatten: Takes an image and a binary mask of a card, detects the
 * card's corners, and applies a perspective transform to "flatten" it into a
 * top-down view.
 * @param image: Input source image (BGR).
 * @param mask: Binary mask where the card is white (CV_8UC1).
 * @return cv::Mat: The flattened card image (approx 200x300 or aspect ratio
 * corrected).
 */
cv::Mat flatten(const cv::Mat &image, const cv::Mat &mask);

/**
 * @brief Get bounding boxes of the two opposite corners of a card (where the number is)
 * in original image coordinates.
 * @param image: Input source image (BGR).
 * @param contour: The contour points defining the card boundary.
 * @param bbox1: Output bounding box for the top-left corner (in original coords).
 * @param bbox2: Output bounding box for the bottom-right corner (in original coords).
 * @param card_projected_image: Output flattened card image (optional, can pass empty Mat).
 * @param outputSize: The size of the projected card (default: 250x350).
 * @param cornerFraction: Fraction of card size for corner region (default: 0.35).
 * @return bool: True if successful, false otherwise.
 */
bool getCornerBboxes(const cv::Mat &image, const std::vector<cv::Point> &contour, cv::Rect &bbox1, cv::Rect &bbox2, cv::Mat &card_projected_image, cv::Size outputSize = cv::Size(250, 350), float cornerFraction = 0.20);

} // namespace CardProjection

#endif // CARDPROJECTION_H

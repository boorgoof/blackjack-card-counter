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

} // namespace CardProjection

#endif // CARDPROJECTION_H

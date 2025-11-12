#ifndef CARDPROJECTION_H
#define CARDPROJECTION_H

#include <opencv2/opencv.hpp>
#include <vector>

namespace CardProjection {
    /**
     * @brief Projects a card contour to a top-down 2D view
     * @param image The source image containing the card
     * @param contour The contour points of the card
     * @return A perspective-corrected image of the card (250x350 pixels)
     */
    cv::Mat projectCard(const cv::Mat& image, const std::vector<cv::Point>& contour);
}

#endif // CARDPROJECTION_H

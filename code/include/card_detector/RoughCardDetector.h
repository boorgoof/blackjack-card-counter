#ifndef ROUGHCARDDETECTOR_H
#define ROUGHCARDDETECTOR_H

#include "../Dataset.h"

/**
 * @class RoughCardDetector
 * @brief A class for detecting playing cards in images using rough detection methods.
 * 
 * The RoughCardDetector class provides methods to detect playing cards in an image by generating masks,
 * extracting polygons, convex hulls, and bounding boxes of the detected cards. It utilizes color thresholding
 * and morphological operations to identify card regions in the image.
 */
class RoughCardDetector{
    public:

        /**
         * @brief Default constructor for the RoughCardDetector class.
         */
        RoughCardDetector();

        /**
        * @brief Generates a mask highlighting the regions in the image that correspond to playing cards.
        *
        * This method processes the input image and returns a binary mask where the detected card regions are marked.
        * The mask can be used for further analysis or extraction of card areas from the original image.
        *
        * @param originalImage The input image in which cards are to be detected.
        * @return cv::Mat A binary mask image with detected card regions.
        */
        cv::Mat getCardsMask(const cv::Mat& originalImage);

        /** 
        * @brief Detects and returns the polygons representing the detected cards in the image.
        *
        * This method processes the input image to identify playing cards and returns their contours as polygons.
        * Each polygon is represented as a vector of cv::Point, which are the vertices of the polygon.
        *
        * @param originalImage The input image in which cards are to be detected.
        * @return std::vector<std::vector<cv::Point>> A vector of polygons, each represented as a vector of cv::Point.
        */
        std::vector<std::vector<cv::Point>> getCardsPolygon(const cv::Mat& originalImage);

        /**
         * @brief Detects and returns the convex hulls of the detected cards in the image.
         * 
         * This method processes the input image to identify playing cards and computes their convex hulls.
         * Each convex hull is represented as a vector of cv::Point, which are the vertices
         * of the convex shape enclosing the card.
         * 
         * @param originalImage The input image in which cards are to be detected.
         * @return std::vector<std::vector<cv::Point>> A vector of convex hulls, each represented as a vector of cv::Point.
         */
        std::vector<std::vector<cv::Point>> getConvexHulls(const cv::Mat& originalImage);

        /**
         * @brief Detects and returns the bounding boxes of the detected cards in the image.
         * 
         * This method processes the input image to identify playing cards and computes their bounding boxes.
         * Each bounding box is represented as a cv::Rect, which defines the rectangle enclosing the card.
         * 
         * @param originalImage The input image in which cards are to be detected.
         * @return std::vector<cv::Rect> A vector of bounding boxes, each represented as a cv::Rect.
         */
        std::vector<cv::Rect> getCardsBoundingBox(const cv::Mat& originalImage);
        
    private:

        /**
         * @brief Applies a white color threshold to the input image to create a binary mask.
         * 
         * This method enhances the input image and converts it to the HSV color space.
         * It then applies a threshold to isolate white regions, which are likely to correspond to playing cards.
         * The resulting mask can be used for further processing and analysis.
         * 
         * @param image The input image to be processed.
         * @return cv::Mat A binary mask image with white regions highlighted.
         */
        cv::Mat whiteTreshold(const cv::Mat& image);

        /**
         * @brief Filters the input mask by size, removing small contours.
         * 
         * This method analyzes the contours in the binary mask and removes those
         * that are smaller than the specified minimum area. This helps to eliminate
         * noise and focus on the larger, more relevant card regions.
         * 
         * @param mask The binary mask image to be processed.
         * @param minArea The minimum area threshold for retaining contours.
         */
        void filterBySize(cv::Mat& mask, int minArea);

        /**
         * @brief Applies morphological operations to clean up the binary mask.
         * 
         * This method performs morphological opening and closing on the input mask
         * to remove noise and fill small holes. The sizes of the structuring elements
         * for opening and closing can be specified.
         * 
         * @param mask The binary mask image to be processed.
         * @param openSize The size of the structuring element for the opening operation.
         * @param closeSize The size of the structuring element for the closing operation.
         */
        void morphologicalCleanup(cv::Mat& mask, int openSize, int closeSize);
};


#endif
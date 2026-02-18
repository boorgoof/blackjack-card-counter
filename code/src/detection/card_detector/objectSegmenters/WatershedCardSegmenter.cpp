// Federico Meneghetti

#include "../../../../include/detection/card_detector/objectSegmenters/WatershedCardSegmenter.h"
#include "../../../../include/ImageFilter.h"

WatershedCardSegmenter::WatershedCardSegmenter()
{
    set_method_name("Watershed (Distance Transform + Markers)");
}

std::vector<std::vector<cv::Point>> WatershedCardSegmenter::segment_objects(const cv::Mat& src_img, const cv::Mat& src_mask)
{
    std::vector<std::vector<cv::Point>> final_contours;

    if (src_img.empty() || src_mask.empty())
        return final_contours;


    // Apply mask
    cv::Mat masked = cv::Mat::zeros(src_img.size(), src_img.type());
    src_img.copyTo(masked, src_mask);

    // preprocessing for watershed segmentation
    // sharpening 
    cv::Mat imgResult = Filters::laplaceFilter(masked);
    // binary image
    cv::Mat bw = Filters::binaryImage(imgResult);
    // distance transform
    cv::Mat dist = Filters::distanceTransformFilter(bw);
    
    // watershed segmentation
    cv::Mat peaks = peaksFromDistance(dist);
    std::vector<std::vector<cv::Point>> marker_contours;
    cv::Mat markers = markersFromPeaks(peaks, marker_contours);
    watershedSegmentation(imgResult, markers, marker_contours);

    
    // Extract contours from watershed result
    for (int label = 1; label <= static_cast<int>(marker_contours.size()); ++label)
    {
        cv::Mat regionMask = (markers == label);
        regionMask.convertTo(regionMask, CV_8U);

        std::vector<std::vector<cv::Point>> contouts;
        cv::findContours(regionMask, contouts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (std::vector<cv::Point> & contour : contouts){
            if (!contour.empty()){
                final_contours.push_back(contour);
            }
        }
                
    }

    return final_contours;
}

/*
//This is the version that follows the documentation, but in this case we prefer to use a single seed for the watershed algorithm, since we are looking for a single card in the image.
cv::Mat WatershedCardSegmenter::peaksFromDistance(const cv::Mat& dist)
{
    cv::Mat peaks = dist.clone();
    // Threshold to obtain the peaks. This will be the markers for the foreground objects
    cv::threshold(peaks, peaks, 0.1, 1.0, cv::THRESH_BINARY);
    
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
    cv::dilate(peaks, peaks, kernel);

    return peaks;
}
*/


// In this case return only the strongest peak
cv::Mat WatershedCardSegmenter::peaksFromDistance(const cv::Mat& dist)
{
     cv::Mat peakMask = cv::Mat::zeros(dist.size(), CV_32F);

    double minDistanceValue, maxDistanceValue;
    cv::Point minDistanceLocation, maxDistanceLocation;

    // Find the global maximum in the distance transform
    cv::minMaxLoc(dist, &minDistanceValue, &maxDistanceValue, &minDistanceLocation, &maxDistanceLocation);
    // Mark the strongest peak
    peakMask.at<float>(maxDistanceLocation) = 1.0f;

    // Dilate the peak to enhance it
    cv::Mat dilationKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
    cv::dilate(peakMask, peakMask, dilationKernel);

    return peakMask;
}


cv::Mat WatershedCardSegmenter::markersFromPeaks(const cv::Mat& peaks, std::vector<std::vector<cv::Point>>& contours)
{
    // Create the CV_8U version of the distance image
    cv::Mat peaks_8u;
    peaks.convertTo(peaks_8u, CV_8U);

    // Find total markers
    cv::findContours(peaks_8u, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // create the marker image for the watershed algorithm
    cv::Mat markers = cv::Mat::zeros(peaks.size(), CV_32S);

    for (size_t i = 0; i < contours.size(); i++)
    {
        cv::drawContours(markers, contours,static_cast<int>(i), cv::Scalar(static_cast<int>(i) + 1), -1);
    }

    // Draw the background marker
    cv::circle(markers, cv::Point(5,5), 3, cv::Scalar(255), -1);
    return markers;
}

cv::Mat WatershedCardSegmenter::watershedSegmentation(const cv::Mat& srcColor, cv::Mat& markers, const std::vector<std::vector<cv::Point>>& marker_contours)
{
    // Perform the watershed algorithm
    cv::watershed(srcColor, markers);

    std::vector<cv::Vec3b> colors;
    for (size_t i = 0; i < marker_contours.size(); i++)
    {
        colors.emplace_back(
            cv::theRNG().uniform(0,256),
            cv::theRNG().uniform(0,256),
            cv::theRNG().uniform(0,256));
    }

    cv::Mat dst = cv::Mat::zeros(markers.size(), CV_8UC3);

    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(marker_contours.size()))
                dst.at<cv::Vec3b>(i,j) = colors[index-1];
        }
    }

    return dst;
}
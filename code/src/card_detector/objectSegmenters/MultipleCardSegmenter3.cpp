#include "../../../include/card_detector/objectSegmenters/MultipleCardSegmenter3.h"
#include <iostream>

MultipleCardSegmenter3::MultipleCardSegmenter3() {
  set_method_name("MultipleCardSegmenter3");
}

std::vector<std::vector<cv::Point>>
MultipleCardSegmenter3::segment_objects(const cv::Mat &src_img,
                                        const cv::Mat &src_mask) {
  // 1. Preprocessing
  cv::Mat mask = src_mask.clone();
  if (mask.channels() > 1) {
    cv::cvtColor(mask, mask, cv::COLOR_BGR2GRAY);
  }

  // Ensure binary mask
  cv::threshold(mask, mask, 127, 255, cv::THRESH_BINARY);

  // Remove noise with morphological opening
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 2);

  // 2. Marker Generation
  // Sure background area
  cv::Mat sure_bg;
  cv::dilate(mask, sure_bg, kernel, cv::Point(-1, -1), 3);

  // Distance transform
  cv::Mat dist_transform;
  cv::distanceTransform(mask, dist_transform, cv::DIST_L2, 5);

  // Normalize for visualization/thresholding logic if needed, but we use
  // absolute values for peaks Find global max to set a relative threshold
  double minVal, maxVal;
  cv::minMaxLoc(dist_transform, &minVal, &maxVal);

  // Threshold to get sure foreground (peaks)
  // We use a relatively high threshold to ensure we get the centers of the
  // cards
  cv::Mat sure_fg;
  cv::threshold(dist_transform, sure_fg, 0.5 * maxVal, 255, 0);
  sure_fg.convertTo(sure_fg, CV_8U);

  // Unknown region
  cv::Mat unknown;
  cv::subtract(sure_bg, sure_fg, unknown);

  // Marker labelling
  int n_labels;
  cv::Mat markers;
  n_labels = cv::connectedComponents(sure_fg, markers);

  // Add one to all labels so that sure background is not 0, but 1
  markers = markers + 1;

  // Mark the region of unknown with zero
  markers.setTo(0, unknown == 255);

  // 3. Watershed
  // Watershed works on 3-channel images
  cv::Mat img_for_watershed = src_img.clone();
  if (img_for_watershed.channels() == 1) {
    cv::cvtColor(img_for_watershed, img_for_watershed, cv::COLOR_GRAY2BGR);
  }

  cv::watershed(img_for_watershed, markers);

  // 4. Corner Extraction
  std::vector<std::vector<cv::Point>> card_corners;

  // Iterate through all labels (skipping 0 which is boundaries, and 1 which is
  // background)
  for (int i = 2; i <= n_labels; i++) {
    cv::Mat object_mask = (markers == i);

    // Find contours of this object
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(object_mask, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    if (!contours.empty()) {
      // Assume the largest contour is the object
      std::vector<cv::Point> c = contours[0];

      // Filter small noise
      if (cv::contourArea(c) < 1000)
        continue;

      // Find rotated bounding box
      cv::RotatedRect rect = cv::minAreaRect(c);

      // Get the 4 corners
      cv::Point2f vertices2f[4];
      rect.points(vertices2f);

      std::vector<cv::Point> vertices;
      for (int j = 0; j < 4; ++j) {
        vertices.push_back(vertices2f[j]);
      }

      card_corners.push_back(vertices);
    }
  }

  // Visualization (Optional, for debugging)

  cv::Mat vis = src_img.clone();
  for (const auto &card : card_corners) {
    for (int j = 0; j < 4; ++j) {
      cv::line(vis, card[j], card[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
    }
  }
  cv::imshow("Segmented Cards", vis);
  cv::waitKey(0);

  return card_corners;
}

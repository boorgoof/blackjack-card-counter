<<<<<<< HEAD:code/src/card_detector/CardProjection.cpp
#include "../../include/card_detector/CardProjection.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
=======
#include "../../../include/detection/card_detector/CardProjection.h"
#include <opencv2/imgproc.hpp>
>>>>>>> e68cb53f36897b1afe3b4075b7e0754354f03e72:code/src/detection/card_detector/CardProjection.cpp

namespace CardProjection {

// Helper to sort points: Top-Left, Top-Right, Bottom-Right, Bottom-Left
static std::vector<cv::Point2f>
sortCorners(const std::vector<cv::Point2f> &corners) {
  std::vector<cv::Point2f> sorted(4);
  std::vector<cv::Point2f> pts = corners;

  // Sort by Y to separate top and bottom
  std::sort(
      pts.begin(), pts.end(),
      [](const cv::Point2f &a, const cv::Point2f &b) { return a.y < b.y; });

  // Top points (first 2) sorted by X
  if (pts[0].x < pts[1].x) {
    sorted[0] = pts[0]; // TL
    sorted[1] = pts[1]; // TR
  } else {
    sorted[0] = pts[1]; // TL
    sorted[1] = pts[0]; // TR
  }

  // Bottom points (last 2) sorted by X
  if (pts[2].x < pts[3].x) {
    sorted[3] = pts[2]; // BL
    sorted[2] = pts[3]; // BR
  } else {
    sorted[3] = pts[3]; // BL
    sorted[2] = pts[2]; // BR
  }
  return sorted;
}

cv::Mat flatten(const cv::Mat &image, const cv::Mat &mask) {
  if (mask.empty() || image.empty()) {
    return cv::Mat();
  }

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  if (contours.empty()) {
    std::cerr << "[CardProjection] No contours found in mask." << std::endl;
    return cv::Mat();
  }

  // Find largest contour (assuming the mask was already filtered, but just in
  // case)
  size_t maxIdx = 0;
  double maxArea = 0;
  for (size_t i = 0; i < contours.size(); ++i) {
    double area = cv::contourArea(contours[i]);
    if (area > maxArea) {
      maxArea = area;
      maxIdx = i;
    }
  }
  const auto &contour = contours[maxIdx];

  // Approximate polygon to get corners
  std::vector<cv::Point2f> corners;
  double perimeter = cv::arcLength(contour, true);
  std::vector<cv::Point> approx;
  cv::approxPolyDP(contour, approx, 0.02 * perimeter, true);

  if (approx.size() == 4) {
    for (const auto &p : approx) {
      corners.push_back(cv::Point2f((float)p.x, (float)p.y));
    }
  } else {
    // Fallback: Use RotatedRect unique corners
    cv::RotatedRect rect = cv::minAreaRect(contour);
    cv::Point2f pts[4];
    rect.points(pts);
    for (int i = 0; i < 4; ++i) {
      corners.push_back(pts[i]);
    }
  }

  // Sort corners
  std::vector<cv::Point2f> srcPoints = sortCorners(corners);

  // Define destination size (Standard playing card ratio ~ 2.5 : 3.5 = 5 : 7)
  // Let's use 400x560 for high quality
  int width = 400;
  int height = 560;

  // Check aspect ratio of source points to determine if card is sideways
  float w1 = cv::norm(srcPoints[0] - srcPoints[1]);
  float h1 = cv::norm(srcPoints[1] - srcPoints[2]);

  // If width > height in source (landscape), and we want portrait output:
  // Rotate points so the short side (TR-BR) becomes the top side (TL-TR)
  if (w1 > h1) {
    std::rotate(srcPoints.begin(), srcPoints.begin() + 1, srcPoints.end());
  }

  // Consistent portrait destination points
  std::vector<cv::Point2f> dstPoints = {{0, 0},
                                        {(float)width, 0},
                                        {(float)width, (float)height},
                                        {0, (float)height}};

  // If the detected card is "wide", we might be mapping a wide rect to a tall
  // rect This effectively rotates it 90 degrees, which is good for storage
  // consistency No special handling needed if we always want portrait output.

  cv::Mat M = cv::getPerspectiveTransform(srcPoints, dstPoints);
  cv::Mat warped;
  cv::warpPerspective(image, warped, M, cv::Size(width, height));

  return warped;
}

} // namespace CardProjection

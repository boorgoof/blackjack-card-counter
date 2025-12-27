#include "../../../include/detection/card_detector/CardProjection.h"
#include <opencv2/imgproc.hpp>

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

  // Find largest contour
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
    cv::RotatedRect rect = cv::minAreaRect(contour);
    cv::Point2f pts[4];
    rect.points(pts);
    for (int i = 0; i < 4; ++i) {
      corners.push_back(pts[i]);
    }
  }

  std::vector<cv::Point2f> srcPoints = sortCorners(corners);

  int width = 400;
  int height = 560;

  float w1 = cv::norm(srcPoints[0] - srcPoints[1]);
  float h1 = cv::norm(srcPoints[1] - srcPoints[2]);

  if (w1 > h1) {
    std::rotate(srcPoints.begin(), srcPoints.begin() + 1, srcPoints.end());
  }

  std::vector<cv::Point2f> dstPoints = {{0, 0},
                                        {(float)width, 0},
                                        {(float)width, (float)height},
                                        {0, (float)height}};

  cv::Mat M = cv::getPerspectiveTransform(srcPoints, dstPoints);
  cv::Mat warped;
  cv::warpPerspective(image, warped, M, cv::Size(width, height));

  return warped;
}

bool getCornerBboxes(const cv::Mat &image,
                     const std::vector<cv::Point> &contour,
                     cv::Rect &bbox1,
                     cv::Rect &bbox2,
                     cv::Mat &card_projected_image,
                     cv::Size outputSize,
                     float cornerFraction) {
  if (contour.size() < 4) {
    std::cerr << "[CardProjection] Contour has less than 4 points." << std::endl;
    return false;
  }

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
    cv::RotatedRect rect = cv::minAreaRect(contour);
    cv::Point2f pts[4];
    rect.points(pts);
    for (int i = 0; i < 4; ++i) {
      corners.push_back(pts[i]);
    }
  }

  // Sort corners: TL, TR, BR, BL
  std::vector<cv::Point2f> srcPoints = sortCorners(corners);

  int width = outputSize.width;
  int height = outputSize.height;

  // Check aspect ratio to handle landscape vs portrait orientation
  float w1 = cv::norm(srcPoints[0] - srcPoints[1]);
  float h1 = cv::norm(srcPoints[1] - srcPoints[2]);

  if (w1 > h1) {
    std::rotate(srcPoints.begin(), srcPoints.begin() + 1, srcPoints.end());
  }

  // Destination points for portrait card
  std::vector<cv::Point2f> dstPoints = {{0, 0},
                                        {(float)width, 0},
                                        {(float)width, (float)height},
                                        {0, (float)height}};

  // Get perspective transform and warp
  cv::Mat H = cv::getPerspectiveTransform(srcPoints, dstPoints);
  cv::warpPerspective(image, card_projected_image, H, outputSize, 
                      cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

  // Inverse transform to map back to original coordinates
  cv::Mat H_inv = H.inv();

  // Define corner regions (where card number is) in projected card space
  int cornerW = static_cast<int>(width * cornerFraction);
  int cornerH = static_cast<int>(height * cornerFraction);

  // Top-left corner (in projected space)
  std::vector<cv::Point2f> topLeftCorner = {
    cv::Point2f(0, 0),
    cv::Point2f((float)cornerW, 0),
    cv::Point2f((float)cornerW, (float)cornerH),
    cv::Point2f(0, (float)cornerH)
  };

  // Bottom-right corner (in projected space)
  std::vector<cv::Point2f> bottomRightCorner = {
    cv::Point2f((float)(width - cornerW), (float)(height - cornerH)),
    cv::Point2f((float)width, (float)(height - cornerH)),
    cv::Point2f((float)width, (float)height),
    cv::Point2f((float)(width - cornerW), (float)height)
  };

  // Transform back to original image coordinates
  std::vector<cv::Point2f> topLeftOriginal, bottomRightOriginal;
  cv::perspectiveTransform(topLeftCorner, topLeftOriginal, H_inv);
  cv::perspectiveTransform(bottomRightCorner, bottomRightOriginal, H_inv);

  // Get bounding rects in original image coords
  bbox1 = cv::boundingRect(topLeftOriginal);
  bbox2 = cv::boundingRect(bottomRightOriginal);

  return true;
}

} // namespace CardProjection

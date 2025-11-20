#ifndef MULTIPLECARDSEGMENTER3_H
#define MULTIPLECARDSEGMENTER3_H

#include "ObjectSegmenter.h"
#include <opencv2/opencv.hpp>
#include <vector>

class MultipleCardSegmenter3 : public ObjectSegmenter {
public:
  MultipleCardSegmenter3();
  ~MultipleCardSegmenter3() override = default;

  std::vector<std::vector<cv::Point>>
  segment_objects(const cv::Mat &src_img, const cv::Mat &src_mask) override;

private:
  // Helper to find local maxima in distance transform
  std::vector<cv::Point> findLocalMaxima(const cv::Mat &dist, float threshold,
                                         int minDistance);
};

#endif // MULTIPLECARDSEGMENTER3_H

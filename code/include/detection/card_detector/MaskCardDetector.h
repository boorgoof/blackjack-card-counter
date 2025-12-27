#ifndef MASK_CARD_DETECTOR_H
#define MASK_CARD_DETECTOR_H

#include <functional>
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief Predefined pipeline configurations for common use cases
 *
 * NONE: Empty pipeline - user must configure manually
 * DEFAULT: Standard pipeline: HSV threshold → Filter by size → Morphology
 * LOW_LIGHT: Optimized for low lighting conditions
 * HIGH_LIGHT: Optimized for high lighting conditions
 * SINGLE_CARD: Optimized for single card detection
 */
enum class PipelinePreset { NONE, DEFAULT, LOW_LIGHT, HIGH_LIGHT, SINGLE_CARD };

/**
 * @brief Mask types for different card detection representations
 *
 * POLYGON: Returns mask filled with detected card polygons
 * CONVEX_HULL: Returns mask filled with convex hulls of detected cards
 * BOUNDING_BOX: Returns mask filled with bounding boxes of detected cards
 */
enum class MaskType { POLYGON, CONVEX_HULL, BOUNDING_BOX };

/**
 * @brief MaskCardDetector: Detects card-like objects in an image using a customizable pipeline.
 * 
 * The pipeline consists of a series of image processing steps, each taking a cv::Mat
 * and returning a cv::Mat. Users can either:
 * 1. Use a predefined pipeline preset (DEFAULT, LOW_LIGHT, HIGH_LIGHT)
 * 2. Build a custom pipeline manually using add_step()
 * 3. Start with a preset and add steps as needed
 */
class MaskCardDetector {
private:
  /**
   * @brief steps_: A vector of functions representing the processing steps in
   * the pipeline.
   */
  std::vector<std::function<cv::Mat(const cv::Mat &)>> steps_;

  /**
   * @brief maskType_: The type of mask to generate when getMask() is called.
   */
  std::function<cv::Mat(const cv::Mat &)> maskType_;

  /**
   * @brief applyPipeline: Applies the entire pipeline to an input image.
   * @param img: Input image to process through the pipeline.
   * @return cv::Mat: Processed image after applying all pipeline steps.
   */
  cv::Mat applyPipeline(const cv::Mat &img) const;

public:

  /**
    * @brief Constructor: Initializes the pipeline based on the specified preset.
    * @param preset: Predefined pipeline configuration (default: DEFAULT)
    * @param maskType: Type of mask to generate when getMask() is called (default: CONVEX_HULL)
    * 
    * Available presets:
    * - NONE: Empty pipeline, requires manual configuration
    * - DEFAULT: HSV threshold → Size filter (2000) → Morphology (5,9)
    * - LOW_LIGHT: HSV threshold (lo={0,0,80}, hi={179,60,255}, alpha=2.0, beta=40.0) → Size filter (2000) → Morphology (5,9)
    * - HIGH_LIGHT: HSV threshold (lo={0,0,140}, hi={179,40,255}, alpha=1.2, beta=10.0) → Size filter (2000) → Morphology (5,9)
    */
  MaskCardDetector(PipelinePreset preset = PipelinePreset::DEFAULT, MaskType maskType = MaskType::CONVEX_HULL);

  /**
   * @brief loadPreset: Replaces the current pipeline with a predefined preset.
   * @param preset: The preset configuration to load
   */
  void loadPreset(PipelinePreset preset);

  /**
   * @brief clearPipeline: Removes all steps from the pipeline.
   */
  void clearPipeline() { steps_.clear(); }

  /**
   * @brief getPipelineSize: Returns the number of steps in the current
   * pipeline.
   * @return size_t: Number of processing steps
   */
  size_t getPipelineSize() const { return steps_.size(); }

  /**
   * @brief isEmpty: Checks if the pipeline is empty.
   * @return bool: True if no processing steps are configured
   */
  bool isEmpty() const { return steps_.empty(); }

  /**
   * @brief add_step: Adds a new processing step to the pipeline.
   * @param fn: A callable that takes a cv::Mat and returns a cv::Mat.
   * @param args: Additional arguments to bind to the callable. These arguments
   * are forwarded and captured by the pipeline step, allowing customization of
   * processing functions with specific parameters.
   *
   * @details The args parameter enables you to create customized pipeline steps
   * by binding additional parameters to processing functions.
   */
  template <typename StepFn, typename... Args>
  void add_step(StepFn fn, Args &&...args) {
    auto wrapped = [fn, ... a = std::forward<Args>(args)](const cv::Mat &in) {
      return fn(in, a...);
    };
    steps_.push_back(std::move(wrapped));
  }

  /**
   * @brief getMask: Unified method to get a mask of detected cards using the
   * mask type specified in constructor.
   * @param img: Input BGR image (cv::Mat).
   * @return cv::Mat: Binary mask where detected card regions are white (255)
   * and background is black (0) (CV_8UC1).
   */
  cv::Mat getMask(const cv::Mat &img) const { return maskType_(img); }

  /**
   * @brief setMaskType: Changes the mask type used by getMask().
   * @param maskType: New mask type to use (POLYGON, CONVEX_HULL, or
   * BOUNDING_BOX).
   */
  void setMaskType(std::function<cv::Mat(const cv::Mat &)> func) {
    maskType_ = func;
  }

  void loadMaskPreset(MaskType maskType);
};

namespace preprocessing {

/**
 * @brief hsvWhiteThreshold: Applies HSV thresholding to isolate white regions
 * in the image.
 * @param bgr: Input BGR image (cv::Mat).
 * @param lo: Lower bound for HSV thresholding (default: {0, 0, 180}).
 * @param hi: Upper bound for HSV thresholding (default: {179, 20, 255}).
 * @param alpha: Contrast adjustment factor (default: 1.2).
 * @param beta: Brightness adjustment factor (default: 10.0).
 * @return cv::Mat: Binary mask where white regions are 255 and others are 0.
 */
cv::Mat hsvWhiteThreshold(const cv::Mat &bgr,
                          const cv::Scalar &lo = {0, 0, 180},
                          const cv::Scalar &hi = {179, 20, 255},
                          double alpha = 1.2, double beta = 10.0);

/**
 * @brief filterBySize: Filters out small contours from a binary mask based on a
 * minimum area threshold.
 * @param maskIn: Input binary mask (cv::Mat).
 * @param minArea: Minimum area threshold to retain a contour (default: 2000).
 * @return cv::Mat: Binary mask with small contours removed.
 */
cv::Mat filterBySize(const cv::Mat &maskIn, int minArea = 2000);

/**
 * @brief morphOpenClose: Applies morphological opening followed by closing to a
 * binary mask.
 * @param maskIn: Input binary mask (cv::Mat).
 * @param openSize: Kernel size for morphological opening (default: 5).
 * @param closeSize: Kernel size for morphological closing (default: 9).
 * @return cv::Mat: Processed binary mask after morphological operations.
 */
cv::Mat morphOpenClose(const cv::Mat &maskIn, int openSize = 5,
                       int closeSize = 9);

/**
 * @brief keepLargestObject: Filters the mask to keep only the largest contour.
 * @param maskIn: Input binary mask (cv::Mat).
 * @return cv::Mat: Binary mask with only the largest object filled.
 */
cv::Mat keepLargestObject(const cv::Mat &maskIn);

} // namespace preprocessing

namespace mask {

/**
 * @brief getCardsPolygon: Extracts the polygonal contours from a binary mask.
 * @param mask: Input binary mask (CV_8UC1).
 * @return std::vector<std::vector<cv::Point>>: List of polygons representing
 * detected cards.
 */
std::vector<std::vector<cv::Point>> getCardsPolygon(const cv::Mat &mask);

/**
 * @brief getCardsConvexHulls: Extracts the convex hulls from detected polygons.
 * @param mask: Input binary mask (CV_8UC1).
 * @return std::vector<std::vector<cv::Point>>: List of convex hulls
 * representing detected cards.
 */
std::vector<std::vector<cv::Point>> getCardsConvexHulls(const cv::Mat &mask);

/**
 * @brief getCardsConvexHullsMask: Creates a mask filled with convex hulls from
 * input mask.
 * @param mask: Input binary mask (CV_8UC1).
 * @return cv::Mat: Binary mask with convex hulls filled.
 */
cv::Mat getCardsConvexHullsMask(const cv::Mat &mask);

/**
 * @brief getCardsBoundingBox: Extracts bounding boxes from detected polygons.
 * @param mask: Input binary mask (CV_8UC1).
 * @return std::vector<cv::Rect>: List of bounding boxes representing detected
 * cards.
 */
std::vector<cv::Rect> getCardsBoundingBox(const cv::Mat &mask);

/**
 * @brief getBoundingBoxesMask: Creates a mask filled with bounding boxes from
 * input mask.
 * @param mask: Input binary mask (CV_8UC1).
 * @return cv::Mat: Binary mask with bounding boxes filled.
 */
cv::Mat getBoundingBoxesMask(const cv::Mat &mask);

} // namespace mask
#endif

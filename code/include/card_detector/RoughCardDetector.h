#ifndef ROUGHCARDDETECTOR_H
#define ROUGHCARDDETECTOR_H

#include <opencv2/opencv.hpp>
#include <functional>
#include <vector>

namespace vision {

/**
 * @brief Predefined pipeline configurations for common use cases
 * 
 * NONE: Empty pipeline - user must configure manually
 * DEFAULT: Standard pipeline: HSV threshold → Filter by size → Morphology
 * LOW_LIGHT: Optimized for low lighting conditions
 * HIGH_LIGHT: Optimized for high lighting conditions
 */
enum class PipelinePreset {
    NONE,
    DEFAULT,
    LOW_LIGHT,
    HIGH_LIGHT 
};

/**
 * @brief Mask types for different card detection representations
 * 
 * POLYGON: Returns mask filled with detected card polygons
 * CONVEX_HULL: Returns mask filled with convex hulls of detected cards
 * BOUNDING_BOX: Returns mask filled with bounding boxes of detected cards
 */
enum class MaskType {
    POLYGON,
    CONVEX_HULL,
    BOUNDING_BOX
};

/**
 * @brief RoughCardDetector: Detects card-like objects in an image using a customizable pipeline.
 * 
 * The pipeline consists of a series of image processing steps, each taking a cv::Mat
 * and returning a cv::Mat. Users can either:
 * 1. Use a predefined pipeline preset (DEFAULT, LOW_LIGHT, HIGH_LIGHT)
 * 2. Build a custom pipeline manually using add_step()
 * 3. Start with a preset and add steps as needed
 */
class RoughCardDetector {
private:

    /**
     * @brief steps_: A vector of functions representing the processing steps in the pipeline.
     */
    std::vector<std::function<cv::Mat(const cv::Mat&)>> steps_;

    /**
     * @brief maskType_: The type of mask to generate when getMask() is called.
     */
    MaskType maskType_;

    /**
     * @brief getCardsPolygon: Extracts the polygonal contours of detected card-like objects.
     * Uses the pipeline to process the image and then finds contours from the resulting binary mask.
     * @param img: Input BGR image (cv::Mat).
     * @return std::vector<std::vector<cv::Point>>: List of polygons representing detected cards.
     */
    std::vector<std::vector<cv::Point>> getCardsPolygon(const cv::Mat& img) const;

        /**
     * @brief getCardPolygonMask: Processes the input image through the pipeline and returns a binary mask
     * where detected card-like objects are white (255) and the background is black (0).
     * @param img: Input BGR image (cv::Mat).
     * @return cv::Mat: Binary mask of detected cards (CV_8UC1).
     */
    cv::Mat getCardPolygonMask(const cv::Mat& img) const;

    /**
     * @brief getCardsConvexHulls: Extracts the convex hulls of detected card-like objects.
     * @param img: Input BGR image (cv::Mat).
     * @return std::vector<std::vector<cv::Point>>: List of convex hulls representing detected cards.
     */
    std::vector<std::vector<cv::Point>> getCardsConvexHulls(const cv::Mat& img) const;

    /**
     * @brief getCardsConvexHullsMask: Processes the input image through the pipeline and returns a binary mask
     * where the convex hulls of detected card-like objects are white (255) and the background is black (0).
     * @param img: Input BGR image (cv::Mat).
     * @return cv::Mat: Binary mask of convex hulls of detected cards (CV_8UC1).
     */
    cv::Mat getCardsConvexHullsMask(const cv::Mat& img) const;

    /**
     * @brief getCardsBoundingBox: Extracts the bounding boxes of detected card-like objects.
     * @param img: Input BGR image (cv::Mat).
     * @return std::vector<cv::Rect>: List of bounding boxes representing detected cards.
     */
    std::vector<cv::Rect> getCardsBoundingBox(const cv::Mat& img) const;

    /**
     * @brief getBoundingBoxesMask: Processes the input image through the pipeline and returns a binary mask
     * where the bounding boxes of detected card-like objects are white (255) and the background is black (0).
     * @param img: Input BGR image (cv::Mat).
     * @return cv::Mat: Binary mask of bounding boxes of detected cards (CV_8UC1).
     */
    cv::Mat getBoundingBoxesMask(const cv::Mat& img) const;


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
    RoughCardDetector(PipelinePreset preset = PipelinePreset::DEFAULT, MaskType maskType = MaskType::CONVEX_HULL);

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
     * @brief getPipelineSize: Returns the number of steps in the current pipeline.
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
     * @param args: Additional arguments to bind to the callable. These arguments are
     *              forwarded and captured by the pipeline step, allowing customization
     *              of processing functions with specific parameters.
     * 
     * @details The args parameter enables you to create customized pipeline steps by
     *          binding additional parameters to processing functions.
     */
    template <typename StepFn, typename... Args>
    void add_step(StepFn fn, Args&&... args) {
        auto wrapped = [fn, ... a = std::forward<Args>(args)](const cv::Mat& in) {
            return fn(in, a...);
        };
        steps_.push_back(std::move(wrapped));
    }

    
    /**
     * @brief getMask: Unified method to get a mask of detected cards using the mask type specified in constructor.
     * @param img: Input BGR image (cv::Mat).
     * @return cv::Mat: Binary mask where detected card regions are white (255) and background is black (0) (CV_8UC1).
     */
    cv::Mat getMask(const cv::Mat& img) const;

    /**
     * @brief setMaskType: Changes the mask type used by getMask().
     * @param maskType: New mask type to use (POLYGON, CONVEX_HULL, or BOUNDING_BOX).
     */
    void setMaskType(MaskType maskType) { maskType_ = maskType; }

    /**
     * @brief getMaskType: Returns the current mask type.
     * @return MaskType: Current mask type setting.
     */
    MaskType getMaskType() const { return maskType_; }
};

} 

namespace {

/**
 * @brief hsvWhiteThreshold: Applies HSV thresholding to isolate white regions in the image.
 * @param bgr: Input BGR image (cv::Mat).
 * @param lo: Lower bound for HSV thresholding (default: {0, 0, 180}).
 * @param hi: Upper bound for HSV thresholding (default: {179, 20, 255}).
 * @param alpha: Contrast adjustment factor (default: 1.2).
 * @param beta: Brightness adjustment factor (default: 10.0).
 * @return cv::Mat: Binary mask where white regions are 255 and others are 0.
 */
cv::Mat hsvWhiteThreshold(const cv::Mat& bgr, cv::Scalar lo = {0, 0, 180}, cv::Scalar hi = {179, 20, 255}, double alpha = 1.2, double beta = 10.0);

/**
 * @brief filterBySize: Filters out small contours from a binary mask based on a minimum area threshold.
 * @param maskIn: Input binary mask (cv::Mat).
 * @param minArea: Minimum area threshold to retain a contour (default: 2000).
 * @return cv::Mat: Binary mask with small contours removed.
 */
cv::Mat filterBySize(const cv::Mat& maskIn, int minArea = 2000);

/**
 * @brief morphOpenClose: Applies morphological opening followed by closing to a binary mask.
 * @param maskIn: Input binary mask (cv::Mat).
 * @param openSize: Kernel size for morphological opening (default: 5).
 * @param closeSize: Kernel size for morphological closing (default: 9).
 * @return cv::Mat: Processed binary mask after morphological operations.
 */
cv::Mat morphOpenClose(const cv::Mat& maskIn, int openSize = 5, int closeSize = 9);

}
#endif

// Gianluca Caregnato

#ifndef SEQUENTIAL_FRAMES_PROCESSING_H
#define SEQUENTIAL_FRAMES_PROCESSING_H

#include "ProcessingMode.h"
#include "card_detector/YoloCardDetector.h"
#include "CardTracker.h"
#include <memory>

/**
 * @brief Processes video frames sequentially, tracking cards across frames
 *        and maintaining a Hi-Lo running count.
 */
class SequentialFrameProcessing : public ProcessingMode {
public:
    /**
     * @brief Construct a sequential frame processor.
     * @param card_detector Card detector to use (ownership transferred).
     * @param visualize     Whether to visualize detections.
     * @param fps           Frames per second of the video source.
     */
    SequentialFrameProcessing(std::unique_ptr<CardDetector> card_detector, bool visualize, double fps = 60.0);
    ~SequentialFrameProcessing();
    
    /**
     * @brief Detect cards in a frame and update the tracker.
     * @param image Input BGR frame.
     * @return Labels detected in this frame.
     */
    std::vector<Label> detect_image(const cv::Mat& image) override;

    void set_fps(double fps) { tracker_.set_fps(fps); }
    double get_fps() const { return tracker_.get_fps(); }
    
    int get_running_count() const { return tracker_.get_running_count(); }
    std::vector<CardType> get_removed_cards_this_frame() const { return tracker_.get_removed_cards_this_frame(); }
    const std::map<std::string, CardTracker::TrackedCard>& get_tracked_cards() const { return tracker_.get_tracked_cards(); }
    const std::set<std::string>& get_background_cards() const { return tracker_.get_background_cards(); }
    void reset_tracking() { tracker_.reset(); }

private:
    std::unique_ptr<CardDetector> card_detector_;
    CardTracker tracker_;
};

#endif

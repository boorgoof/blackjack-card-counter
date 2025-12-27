#ifndef SEQUENTIAL_FRAMES_PROCESSING_H
#define SEQUENTIAL_FRAMES_PROCESSING_H

#include "ProcessingMode.h"
#include "card_detector/YoloCardDetector.h"
#include "CardTracker.h"
#include <memory>

/**
 * Processes video frames sequentially, tracking cards across frames
 * and maintaining a Hi-Lo count for card counting.
 */
class SequentialFrameProcessing : public ProcessingMode {
public:
    SequentialFrameProcessing(double fps = 1.0, bool detect_full_card = false, bool visualize = false);
    ~SequentialFrameProcessing();
    
    std::vector<Label> detect_image(const cv::Mat& image) override;

    void set_model_path(const std::string& path);
    void set_fps(double fps) { tracker_.set_fps(fps); }
    double get_fps() const { return tracker_.get_fps(); }
    
    int get_running_count() const { return tracker_.get_running_count(); }
    std::vector<CardType> get_removed_cards_this_frame() const { return tracker_.get_removed_cards_this_frame(); }
    const std::map<std::string, CardTracker::TrackedCard>& get_tracked_cards() const { return tracker_.get_tracked_cards(); }
    const std::set<std::string>& get_background_cards() const { return tracker_.get_background_cards(); }
    void reset_tracking() { tracker_.reset(); }

private:
    std::string model_path_ = "../DL_approach/yolov11s_synthetic_1280.onnx";
    std::unique_ptr<YoloCardDetector> card_detector_;
    CardTracker tracker_;
    
    void init_detector();
};

#endif

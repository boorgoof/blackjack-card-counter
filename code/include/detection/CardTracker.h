#ifndef CARD_TRACKER_H
#define CARD_TRACKER_H

#include <map>
#include <set>
#include <vector>
#include <string>
#include "../Label.h"
#include "../CardType.h"

/**
 * Tracks cards across video frames for Hi-Lo counting.
 * 
 * All timing thresholds are in SECONDS and converted to frames using FPS.
 * 
 * Key insight: Each physical card has 2 visible labels (corners).
 * So we count by card ID and divide by 2 to get actual card count:
 * - 1-2 detections = 1 physical card
 * - 3-4 detections = 2 physical cards (multi-deck)
 * 
 * State machine per card ID:
 * - CANDIDATE: Recently appeared, not yet confirmed
 * - CONFIRMED: Stable on table
 * - OCCLUDED: Was confirmed but temporarily not detected
 * - BACKGROUND: Detected too many seconds, ignored
 */
class CardTracker {
public:
    // Timing thresholds in SECONDS
    static constexpr double SECONDS_TO_CONFIRM = 3.0;
    static constexpr double SECONDS_OCCLUSION = 3.0;
    static constexpr double SECONDS_BACKGROUND = 30.0;

    enum class CardState { CANDIDATE, CONFIRMED, OCCLUDED, BACKGROUND };

    struct TrackedCard {
        CardType card;
        int detection_count;          // Labels detected this frame (1 or 2 per physical card)
        int confirmed_card_count;     // = ceil(detection_count / 2)
        int frames_detected;          // Frames this card has been seen
        int frames_since_last_seen;
        CardState state;

        TrackedCard(const CardType& c, int det_count) 
            : card(c), detection_count(det_count), 
              confirmed_card_count((det_count + 1) / 2),
              frames_detected(1), frames_since_last_seen(0), 
              state(CardState::CANDIDATE) {}
    };

    CardTracker(double fps = 1.0);
    ~CardTracker() = default;

    void set_fps(double fps);
    double get_fps() const { return fps_; }
    
    void update_frame(const std::vector<Label>& detections);
    std::vector<CardType> get_removed_cards_this_frame() const;
    int get_cards_removed_count_this_frame() const;
    int get_running_count() const { return running_count_; }
    const std::map<std::string, TrackedCard>& get_tracked_cards() const { return tracked_cards_; }
    const std::set<std::string>& get_background_cards() const { return background_cards_; }
    void reset();

private:
    double fps_ = 1.0;
    int frames_to_confirm_ = 3;
    int frames_occlusion_ = 3;
    int frames_background_ = 30;
    
    std::map<std::string, TrackedCard> tracked_cards_;
    std::set<std::string> background_cards_;
    std::vector<CardType> removed_this_frame_;
    int cards_removed_count_this_frame_ = 0;
    int running_count_ = 0;

    void update_frame_thresholds();
    static const CardType* extract_card(const Label& label);
};

#endif

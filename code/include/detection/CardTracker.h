// Gianluca Caregnato

#ifndef CARD_TRACKER_H
#define CARD_TRACKER_H

#include <map>
#include <set>
#include <vector>
#include <string>
#include "../Label.h"
#include "../CardType.h"

/**
 * @brief Tracks cards across video frames for Hi-Lo counting.
 *
 * All timing thresholds are in SECONDS and converted to frames using FPS.
 *
 * State machine per card ID:
 * - CANDIDATE: Recently appeared, not yet confirmed
 * - CONFIRMED: Stable on table
 * - OCCLUDED: Was confirmed but temporarily not detected
 * - BACKGROUND: Detected too many seconds, ignored
 */
class CardTracker {
public:
    static constexpr double SECONDS_TO_CONFIRM = 2.0;
    static constexpr double SECONDS_OCCLUSION = 3.0;
    static constexpr double SECONDS_BACKGROUND = 30.0;

    enum class CardState { CANDIDATE, CONFIRMED, OCCLUDED, BACKGROUND };

    struct TrackedCard {
        CardType card;
        int detection_count;
        int confirmed_card_count;
        int frames_detected;
        int frames_since_last_seen;
        CardState state;

        TrackedCard(const CardType& c, int det_count) 
            : card(c), detection_count(det_count), 
              confirmed_card_count((det_count + 1) / 2),
              frames_detected(1), frames_since_last_seen(0), 
              state(CardState::CANDIDATE) {}
    };

    /**
     * @brief Construct a CardTracker.
     * @param fps Frames per second of the video source.
     */
    CardTracker(double fps = 1.0);
    ~CardTracker() = default;

    /**
     * @brief Set the FPS and recalculate frame thresholds.
     * @param fps New frames-per-second value.
     */
    void set_fps(double fps);
    double get_fps() const { return fps_; }

    /**
     * @brief Process a new frame's detections, updating all tracked card states.
     * @param detections Labels detected in the current frame.
     */
    void update_frame(const std::vector<Label>& detections);

    /**
     * @brief Get the cards that were removed (left the table) this frame.
     * @return Vector of CardType objects removed this frame.
     */
    std::vector<CardType> get_removed_cards_this_frame() const;

    /**
     * @brief Get how many cards were removed this frame.
     * @return Number of cards removed.
     */
    int get_cards_removed_count_this_frame() const;

    int get_running_count() const { return running_count_; }
    const std::map<std::string, TrackedCard>& get_tracked_cards() const { return tracked_cards_; }
    const std::set<std::string>& get_background_cards() const { return background_cards_; }

    /**
     * @brief Reset all tracking state (tracked cards, background, running count).
     */
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

    /**
     * @brief Recalculate frame thresholds from the current FPS.
     */
    void update_frame_thresholds();

    /**
     * @brief Recalculate the Hi-Lo running count from all confirmed/occluded cards.
     */
    void recalculate_running_count();

    /**
     * @brief Extract a CardType pointer from a Label (returns nullptr if not a card).
     * @param label The label to extract from.
     * @return Pointer to the CardType, or nullptr.
     */
    static const CardType* extract_card(const Label& label);
};

#endif

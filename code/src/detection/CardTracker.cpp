#include "../../include/detection/CardTracker.h"
#include <cmath>

CardTracker::CardTracker(double fps) : fps_(fps) {
    update_frame_thresholds();
}

void CardTracker::set_fps(double fps) {
    fps_ = fps;
    update_frame_thresholds();
}

void CardTracker::update_frame_thresholds() {
    frames_to_confirm_ = static_cast<int>(std::ceil(SECONDS_TO_CONFIRM * fps_));
    frames_occlusion_ = static_cast<int>(std::ceil(SECONDS_OCCLUSION * fps_));
    frames_background_ = static_cast<int>(std::ceil(SECONDS_BACKGROUND * fps_));
    
    // Minimum 1 frame for each threshold
    if (frames_to_confirm_ < 1) frames_to_confirm_ = 1;
    if (frames_occlusion_ < 1) frames_occlusion_ = 1;
    if (frames_background_ < 1) frames_background_ = 1;
}

void CardTracker::update_frame(const std::vector<Label>& detections) {
    removed_this_frame_.clear();
    cards_removed_count_this_frame_ = 0;

    // Count detections per card ID this frame
    std::map<std::string, int> detection_counts;
    std::map<std::string, const CardType*> card_ptrs;
    
    for (size_t i = 0; i < detections.size(); ++i) {
        const Label& label = detections[i];
        const CardType* card = extract_card(label);
        if (card && card->isValid()) {
            std::string id = card->get_id();
            if (background_cards_.find(id) == background_cards_.end()) {
                detection_counts[id]++;
                card_ptrs[id] = card;
            }
        }
    }

    // Update existing tracked cards
    std::vector<std::string> to_remove;
    
    for (std::map<std::string, TrackedCard>::iterator it = tracked_cards_.begin();
         it != tracked_cards_.end(); ++it) {
        std::string card_id = it->first;
        TrackedCard& tracked = it->second;
        
        std::map<std::string, int>::iterator det_it = detection_counts.find(card_id);
        
        if (det_it != detection_counts.end()) {
            // Card detected this frame
            int det_count = det_it->second;
            tracked.detection_count = det_count;
            tracked.confirmed_card_count = (det_count + 1) / 2;
            tracked.frames_detected++;
            tracked.frames_since_last_seen = 0;

            // Check for background card
            if (tracked.frames_detected > frames_background_) {
                tracked.state = CardState::BACKGROUND;
                background_cards_.insert(card_id);
                to_remove.push_back(card_id);
            }
            else if (tracked.state == CardState::CANDIDATE) {
                if (tracked.frames_detected >= frames_to_confirm_) {
                    tracked.state = CardState::CONFIRMED;
                }
            } 
            else if (tracked.state == CardState::OCCLUDED) {
                tracked.state = CardState::CONFIRMED;
            }

            detection_counts.erase(det_it);
        } 
        else {
            // Card NOT detected this frame
            tracked.frames_since_last_seen++;

            if (tracked.state == CardState::CONFIRMED) {
                tracked.state = CardState::OCCLUDED;
            } 
            else if (tracked.state == CardState::OCCLUDED) {
                if (tracked.frames_since_last_seen > frames_occlusion_) {
                    // Card(s) left the table
                    int num_cards = tracked.confirmed_card_count;
                    for (int c = 0; c < num_cards; ++c) {
                        removed_this_frame_.push_back(tracked.card);
                        Blackjack::HiLo value = Blackjack::rank_to_HiLo(tracked.card.get_rank());
                        running_count_ += Blackjack::HiLo_to_int(value);
                    }
                    cards_removed_count_this_frame_ += num_cards;
                    to_remove.push_back(card_id);
                }
            } 
            else if (tracked.state == CardState::CANDIDATE) {
                if (tracked.frames_since_last_seen > frames_occlusion_) {
                    to_remove.push_back(card_id);
                }
            }
        }
    }

    // Remove processed cards
    for (size_t i = 0; i < to_remove.size(); ++i) {
        tracked_cards_.erase(to_remove[i]);
    }

    // Add remaining detections as new tracked cards
    for (std::map<std::string, int>::const_iterator it = detection_counts.begin();
         it != detection_counts.end(); ++it) {
        std::string card_id = it->first;
        int det_count = it->second;
        const CardType* card_ptr = card_ptrs[card_id];
        if (card_ptr) {
            tracked_cards_.insert(std::make_pair(card_id, TrackedCard(*card_ptr, det_count)));
        }
    }
}

std::vector<CardType> CardTracker::get_removed_cards_this_frame() const {
    return removed_this_frame_;
}

int CardTracker::get_cards_removed_count_this_frame() const {
    return cards_removed_count_this_frame_;
}

void CardTracker::reset() {
    tracked_cards_.clear();
    background_cards_.clear();
    removed_this_frame_.clear();
    cards_removed_count_this_frame_ = 0;
    running_count_ = 0;
}

const CardType* CardTracker::extract_card(const Label& label) {
    const ObjectType* obj = label.get_object();
    if (!obj) return nullptr;
    return dynamic_cast<const CardType*>(obj);
}

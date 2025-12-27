#include <iostream>
#include <opencv2/opencv.hpp>
#include "../include/Dataset/Dataset.h"
#include "../include/VideoWriter.h"
#include "../include/Dataset/ImageDataset.h"
#include "../include/Dataset/VideoDataset.h"
#include "../include/SampleInfo/SampleInfo.h"
#include "../include/detection/SequentialFrameProcessing.h"
#include "../include/Utils.h"
#include <memory>
#include <vector>

/**
 * Check if a label is a background card (should not be displayed).
 */
bool is_background_card(const Label& label, const std::set<std::string>& background_cards) {
    const ObjectType* obj = label.get_object();
    if (obj) {
        const CardType* card = dynamic_cast<const CardType*>(obj);
        if (card && background_cards.find(card->get_id()) != background_cards.end()) {
            return true;
        }
    }
    return false;   
}

int main() {
    // Dataset configuration
    const bool use_video_dataset = true;
    std::unique_ptr<Dataset> dataset;
    
    if (use_video_dataset) {
        dataset = std::make_unique<VideoDataset>(std::string("../data/datasets/VideoBlackjack.mp4"), 10.0);
    } else {
        dataset = std::make_unique<ImageDataset>(
            std::string("../data/datasets/single_cards/Images/Images"), 
            std::string("../data/datasets/single_cards/YOLO_Annotations/YOLO_Annotations/"));
    }

    std::cout << "Dataset loaded with " << dataset->size() << " frames" << std::endl;

    // Initialize processing mode with card tracking (FPS must match video sample rate)
    const double sample_fps = 10.0;
    std::unique_ptr<SequentialFrameProcessing> mode = std::make_unique<SequentialFrameProcessing>(sample_fps);
    VideoWriter videoW("output_video.mp4", 15.0);

    int frame_number = 0;
    for (Dataset::Iterator it = dataset->begin(); it != dataset->end(); ++it) {
        frame_number++;
        cv::Mat img = dataset->load(it);
        
        if (img.empty()) continue;

        // padding from 1280x720 to 1280x1280
        cv::copyMakeBorder(img, img, 280, 280, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        // Detect cards and update tracking
        std::vector<Label> all_labels = mode->detect_image(img);
        const std::set<std::string>& background_cards = mode->get_background_cards();
        
        // Visualization - draw labels except background cards
        cv::Mat output_img = img.clone();
        for (size_t i = 0; i < all_labels.size(); ++i) {
            const Label& label = all_labels[i];
            if (!is_background_card(label, background_cards)) {
                const std::vector<cv::Rect>& bboxes = label.get_bounding_boxes();
                const ObjectType* obj = label.get_object();
                for (size_t j = 0; j < bboxes.size(); ++j) {
                    const cv::Rect& bbox = bboxes[j];
                    cv::rectangle(output_img, bbox, cv::Scalar(0, 255, 0), 2);
                    if (obj) {
                        cv::putText(output_img, obj->to_string(), 
                            cv::Point(bbox.x, bbox.y - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
                    }
                }
            }
        }

        // Display Hi-Lo count
        int count = mode->get_running_count();
        std::string count_text = "Hi-Lo Count: " + std::to_string(count);
        cv::putText(output_img, count_text, cv::Point(20, 50), 
            cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255, 255, 255), 3);
        cv::putText(output_img, count_text, cv::Point(20, 50), 
            cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 2);

        // Display frame number
        cv::putText(output_img, "Frame: " + std::to_string(frame_number), cv::Point(20, 100), 
            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);

        // Display tracked cards status (card ID, state, and physical card count)
        const std::map<std::string, CardTracker::TrackedCard>& tracked_cards = mode->get_tracked_cards();
        int y_offset = 140;
        for (std::map<std::string, CardTracker::TrackedCard>::const_iterator it = tracked_cards.begin();
             it != tracked_cards.end(); ++it) {
            const std::string& card_id = it->first;
            const CardTracker::TrackedCard& tracked = it->second;
            std::string state_str;
            cv::Scalar color;
            switch (tracked.state) {
                case CardTracker::CardState::CANDIDATE: 
                    state_str = "CAND"; color = cv::Scalar(0, 255, 255); break;
                case CardTracker::CardState::CONFIRMED: 
                    state_str = "CONF"; color = cv::Scalar(0, 255, 0); break;
                case CardTracker::CardState::OCCLUDED: 
                    state_str = "OCCL"; color = cv::Scalar(0, 165, 255); break;
                default: continue;
            }
            // Show card ID, state, and count (e.g., "AS [CONF] x1" or "AS [CONF] x2")
            std::string info = card_id + " [" + state_str + "] x" + std::to_string(tracked.confirmed_card_count);
            cv::putText(output_img, info, cv::Point(20, y_offset), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
            y_offset += 25;
        }

        videoW.addFrame(output_img);
        //print the progression of the cicle 
        std::cout << "Frame " << frame_number << " processed" << std::endl;
        //cv::imshow("Blackjack Card Counter", output_img);
        
        //int key = cv::waitKey(0);
        //if (key == 'q' || key == 'Q' || key == 27) break;

    }
    
    // Close and save the video file
    videoW.close();
    std::cout << "\nVideo saved to: output_video.mp4" << std::endl;
    std::cout << "\n=== Final Hi-Lo Count: " << mode->get_running_count() << " ===" << std::endl;
    cv::destroyAllWindows();
    return 0;
}

#include "../include/Utils.h"
#include "../include/CardType.h"

std::string Utils::Path::longestCommonPath(const std::string& path1_str, const std::string& path2_str) {
    std::filesystem::path path1(path1_str);
    std::filesystem::path path2(path2_str);

    return longestCommonPath(path1, path2).string();
}

std::filesystem::path Utils::Path::longestCommonPath(const std::filesystem::path& path1, const std::filesystem::path& path2) {
    auto it1 = path1.begin();
    auto it2 = path2.begin();
    std::filesystem::path common_path;

    while (it1 != path1.end() && it2 != path2.end() && *it1 == *it2) {
        common_path /= *it1;
        ++it1;
        ++it2;
    }

    return common_path;
}

std::string Utils::String::normalize(const std::string& str) {
    
    std::string t = str;  
    
    t.erase(std::remove_if(t.begin(), t.end(), [](unsigned char c){ return std::isspace(c); }), t.end());
    std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c){ return std::toupper(c); });
    
    return t;
}


void Utils::Save::saveLabelsToYoloFile(const std::string &file_path, const std::vector<Label> &labels, const int image_width, const int image_height)
{
    std::ofstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + file_path);
    }

    for (const auto& label : labels) {
        if (!label.get_object()) {
            continue;
        }
        const std::vector<cv::Rect>& bboxes = label.get_bounding_boxes();
        for (const auto& bbox : bboxes) {
            float x_center = (bbox.x + bbox.width / 2.0f) / image_width;
            float y_center = (bbox.y + bbox.height / 2.0f) / image_height;
            float width = static_cast<float>(bbox.width) / image_width;
            float height = static_cast<float>(bbox.height) / image_height;

            file << label.get_object()->get_id_number() << " "
                << x_center << " "
                << y_center << " "
                << width << " "
                << height << "\n";
        }
    }

    file.close();
}

void Utils::Save::saveImageToFile(const std::string &file_path, const cv::Mat &image)
{
    cv::imwrite(file_path, image);
}

void Utils::Save::save_confusion_matrix(const std::string &file_path, const cv::Mat &confusion_matrix)
{
    std::ofstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + file_path);
    }
    for (int i = 0; i < confusion_matrix.rows; ++i) {
        for (int j = 0; j < confusion_matrix.cols; ++j) {
            file << confusion_matrix.at<int>(i, j);
            if (j < confusion_matrix.cols - 1) {
                file << ",";
            }
        }
        file << "\n";
    }
    file.close();
}

void Utils::Save::save_metrics(const std::string& file_path, const float accuracy, const std::vector<float>& precision, const std::vector<float>& recall, const std::vector<float>& f1, const std::set<int>& classes_to_select) 
{
    std::ofstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + file_path);
    }

    file << "Accuracy: " << accuracy << "\n\n";
    file << "Class,Precision,Recall,F1-Score\n";

    for (int cls : classes_to_select) {
        file << Yolo_index_codec::yolo_index_to_card(cls) << "," << precision[cls] << "," << recall[cls] << "," << f1[cls] << "\n";
    }

    file.close();
}

void Utils::Visualization::printProgressBar(float progress, size_t barwidth, const std::string& prefix, const std::string& suffix) {
    static int lines_to_move_up = 0;

    if (lines_to_move_up > 0) {
        std::cout << "\033[" << lines_to_move_up << "A";
    }

    std::cout << "\r\033[J" << prefix << " [";

    size_t pos = static_cast<size_t>(barwidth * progress);
    for (size_t i = 0; i < barwidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }

    std::cout << "] " << int(progress * 100.0) << "% " << suffix << std::flush;

    lines_to_move_up = std::count(suffix.begin(), suffix.end(), '\n');

    if (progress >= 1.0) {
        std::cout << std::endl;
        lines_to_move_up = 0;
    }
}

void Utils::Visualization::showImage(const cv::Mat &image, const std::string &window_name, const int time, const float resize_factor)
{
    //clone the image to avoid modifying the original
    cv::Mat resized_image = image.clone();
    if (resize_factor > 0.0) {
        if(resize_factor != 1.0){
            cv::Size new_size(static_cast<int>(image.cols * resize_factor), static_cast<int>(image.rows * resize_factor));
            showImage(resized_image, window_name, time, new_size);
        }
        else{
            showImage(resized_image, window_name, time, cv::Size());
        }
    }
    else{
        std::cerr << "Invalid resize factor: " << resize_factor << ". It must be greater than 0." << std::endl;
    }
}

void Utils::Visualization::showImage(const cv::Mat &image, const std::string &window_name, const int time, const cv::Size& size)
{
    if (size != cv::Size()) {
        cv::resize(image, image, size);
    }
    cv::imshow(window_name, image);
    cv::waitKey(time);
    cv::destroyAllWindows();
}

void Utils::Visualization::printLabelsOnImage(cv::Mat &image, const std::vector<Label> &labels, const cv::Scalar &box_color, const cv::Scalar &text_color)
{
    for (const auto& label : labels) {
        const std::vector<cv::Rect>& bboxes = label.get_bounding_boxes();
        for (const auto& bbox : bboxes) {
            cv::rectangle(image, bbox, box_color, 2);
            if (label.get_object()) {
                cv::putText(image, label.get_object()->to_string(), cv::Point(bbox.x, bbox.y - 10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2);
            }
        }
    }
}

void Utils::Visualization::printLabelsOnImageHiLo(cv::Mat &image, const std::vector<Label> &labels)
{
    // Hi-Lo color coding (BGR format):
    // Green = +1 (cards 2-6)
    // Blue  =  0 (cards 7-9)
    // Red   = -1 (cards 10, J, Q, K, A)
    const cv::Scalar COLOR_POS(0, 255, 0);      // Green
    const cv::Scalar COLOR_NEUTRAL(255, 150, 0); // Blue
    const cv::Scalar COLOR_NEG(0, 0, 255);       // Red

    for (const auto& label : labels) {
        const std::vector<cv::Rect>& bboxes = label.get_bounding_boxes();
        
        // Determine color based on Hi-Lo value
        cv::Scalar box_color = COLOR_NEUTRAL;  // default
        const ObjectType* obj = label.get_object();
        if (obj) {
            const CardType* card = dynamic_cast<const CardType*>(obj);
            if (card && card->isValid()) {
                Blackjack::HiLo hilo = Blackjack::rank_to_HiLo(card->get_rank());
                int value = Blackjack::HiLo_to_int(hilo);
                if (value > 0) box_color = COLOR_POS;
                else if (value < 0) box_color = COLOR_NEG;
                else box_color = COLOR_NEUTRAL;
            }
        }

        for (const auto& bbox : bboxes) {
            cv::rectangle(image, bbox, box_color, 2);
            if (obj) {
                cv::putText(image, obj->to_string(), cv::Point(bbox.x, bbox.y - 10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2);
            }
        }
    }
}

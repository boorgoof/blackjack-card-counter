#include "../include/Utils.h"

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
        const cv::Rect& bbox = label.get_bounding_box();
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

    file.close();
}

void Utils::Save::saveImageToFile(const std::string &file_path, const cv::Mat &image)
{
    cv::imwrite(file_path, image);
}

void Utils::Visualization::printProgressBar(float progress, size_t barwidth, const std::string& prefix, const std::string& suffix) {
    std::cout << "\r" << prefix << " [";
    size_t pos = static_cast<size_t>(barwidth * progress);
    for (size_t i = 0; i < barwidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% " << suffix << std::flush;
    if (progress >= 1.0) std::cout << std::endl;
}

void Utils::Visualization::showImage(cv::Mat &image, const std::string &window_name, const int time, const float resize_factor)
{
    if (resize_factor != 1.0 && resize_factor > 0.0) {
        cv::Size new_size(static_cast<int>(image.cols * resize_factor), static_cast<int>(image.rows * resize_factor));
        cv::resize(image, image, new_size);
    }
    cv::imshow(window_name, image);
    cv::waitKey(time);
    cv::destroyAllWindows();
}

void Utils::Visualization::showImage(cv::Mat &image, const std::string &window_name, const int time, const cv::Size& size)
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
        cv::rectangle(image, label.get_bounding_box(), box_color, 2);
        if (label.get_object()) {
            cv::putText(image, label.get_object()->to_string(), cv::Point(label.get_bounding_box().x, label.get_bounding_box().y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2);
        }
    }
}

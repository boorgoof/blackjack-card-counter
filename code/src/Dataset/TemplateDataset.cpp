#include "../../include/Dataset/TemplateDataset.h"
#include "../../include/SampleInfo/TemplateInfo.h"
#include <regex>
#include <iostream>

TemplateDataset::TemplateDataset(const std::string &dataset_path)
{
    image_root_ = std::filesystem::path(dataset_path);
    entries_ = build_entries();
}

Dataset::Iterator TemplateDataset::begin() const
{
    return Iterator(entries_.cbegin());
}

Dataset::Iterator TemplateDataset::end() const
{
    return Iterator(entries_.cend());
}

cv::Mat TemplateDataset::load(const Dataset::Iterator& it)
{
    if (entries_.empty() || it == Iterator(entries_.cend())) {
        std::cerr << "TemplateDataset: invalid iterator or empty dataset" << std::endl;
        return {};
    }
    const SampleInfo& sample = *it;
    cv::Mat image = Loader::Image::load_image(sample.get_pathSample());
    if (image.empty()) {
        std::cerr << "TemplateDataset: failed to load image from " << sample.get_pathSample() << std::endl;
    }
    return image;
}

std::vector<std::shared_ptr<SampleInfo>> TemplateDataset::build_entries()
{
    std::vector<std::shared_ptr<SampleInfo>> entries;

    if (!std::filesystem::exists(image_root_) || !std::filesystem::is_directory(image_root_)) {
        std::cerr << "TemplateDataset: image root directory does not exist or is not a directory: " << image_root_ << std::endl;
        return entries;
    }

    entries.reserve(std::distance(std::filesystem::directory_iterator(image_root_), std::filesystem::directory_iterator{}));

    for (const std::filesystem::directory_entry& dirent : std::filesystem::directory_iterator(image_root_)) {
        if (!dirent.is_regular_file()) continue;

        std::string image_path = dirent.path().string();
        std::string name = dirent.path().stem().string();
        std::regex card_regex("([CDHS])((10)|[A23456789TJQK])");
        std::smatch match;
        if (std::regex_search(name, match, card_regex)) {
            CardType card_type = CardType(CardType::string_to_rank(match[2].str()), CardType::string_to_suit(match[1].str()));
            if (!card_type.isValid()) {
                std::cerr << "Unknown card type in template card filename: " << name << std::endl;
                continue;
            }

            entries.push_back(std::make_unique<TemplateInfo>(name, image_path, card_type));
        }
    }

    std::sort(entries.begin(), entries.end(), [](const std::shared_ptr<SampleInfo>& a, const std::shared_ptr<SampleInfo>& b){
        return a->get_name() < b->get_name();
    });

    return entries;
}

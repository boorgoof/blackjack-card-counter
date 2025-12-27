#include "../include/CardType.h"
#include "../include/Label.h"
#include "../include/Utils.h"
#include <algorithm>
#include <cctype>
#include <stdexcept>

   
CardType::CardType(const std::string& card_text) {
    set_type(card_text);
}

void CardType::set_type(const std::string& card_text) {

    if (card_text.size() < 2) {
        rank = Rank::UNKNOWN; 
        suit = Suit::UNKNOWN; 
        return;
    }
    std::string str_norm = Utils::String::normalize(card_text);
    
    // suit = last char
    const char suit_ch = str_norm.back();
    str_norm.pop_back();  
    
    // rank = what remains
    const Rank& rk  = CardType::string_to_rank(str_norm);                 
    const Suit& su = CardType::string_to_suit(std::string(1, suit_ch)); 

    rank = rk;
    suit = su;
}


std::map<std::string, CardType::Rank> CardType::map_string_to_rank = {
    {"K",  CardType::Rank::K},
    {"Q",  CardType::Rank::Q},
    {"J",  CardType::Rank::J},
    {"10", CardType::Rank::R10},
    {"9",  CardType::Rank::R9},
    {"8",  CardType::Rank::R8},
    {"7",  CardType::Rank::R7},
    {"6",  CardType::Rank::R6},
    {"5",  CardType::Rank::R5},
    {"4",  CardType::Rank::R4},
    {"3",  CardType::Rank::R3},
    {"2",  CardType::Rank::R2},
    {"A",  CardType::Rank::A},
    {"UNK", CardType::Rank::UNKNOWN}
};

std::map<std::string, CardType::Suit> CardType::map_string_to_suit = {
    {"C", CardType::Suit::CLUBS},
    {"D", CardType::Suit::DIAMONDS},
    {"H", CardType::Suit::HEARTS},
    {"S", CardType::Suit::SPADES},
    {"UNK", CardType::Suit::UNKNOWN}
};

std::map<CardType::Rank, std::string> CardType::map_rank_to_string = Utils::Map::createInverseMap(CardType::map_string_to_rank);
std::map<CardType::Suit, std::string> CardType::map_suit_to_string = Utils::Map::createInverseMap(CardType::map_string_to_suit);

std::string CardType::get_id() const {

    const std::string r = map_rank_to_string[this->rank];
    const std::string s = map_suit_to_string[this->suit];
    
    return r + s; // for example: "10H", 10 of Hearts
}

int CardType::get_id_number() const{
    return Yolo_index_codec::card_to_yolo_index(this->get_id());
}

const CardType::Rank CardType::string_to_rank(const std::string& s) {
    auto it = map_string_to_rank.find(s);
    return it != map_string_to_rank.end() ? it->second : CardType::Rank::UNKNOWN;
}

const CardType::Suit CardType::string_to_suit(const std::string& s) {
    auto it = map_string_to_suit.find(s);
    return it != map_string_to_suit.end() ? it->second : CardType::Suit::UNKNOWN;
}

card_color_utils::CardColor card_color_utils::suit_to_color(CardType::Suit suit) {
    switch (suit) {
        case CardType::Suit::SPADES:
        case CardType::Suit::CLUBS:
            return card_color_utils::CardColor::BLACK;

        case CardType::Suit::DIAMONDS:
        case CardType::Suit::HEARTS:
            return card_color_utils::CardColor::RED;

        case CardType::Suit::UNKNOWN:
        default:
            return card_color_utils::CardColor::UNKNOWN;
    }
}

std::vector<CardType::Suit> card_color_utils::color_to_suits(card_color_utils::CardColor color) {
    switch (color) {
        case card_color_utils::CardColor::RED:
            return {CardType::Suit::DIAMONDS, CardType::Suit::HEARTS};

        case card_color_utils::CardColor::BLACK:
        default:
            return {CardType::Suit::UNKNOWN, CardType::Suit::UNKNOWN};
    }
}

cv::Scalar card_color_utils::to_scalar(card_color_utils::CardColor c)
{
    switch (c) {
        case card_color_utils::CardColor::BLACK:
            return cv::Scalar(0, 0, 0);      

        case card_color_utils::CardColor::RED:
            return cv::Scalar(0, 0, 255);    

        default:
            return cv::Scalar(0, 0, 0);
    }
}


bool CardType::operator<(const ObjectType& other) const {

    CardType other_card = dynamic_cast<const CardType&>(other);
    
    if (this->get_suit() != other_card.get_suit())
        return this->get_suit() < other_card.get_suit();
    return this->get_rank() < other_card.get_rank();
}

bool CardType::operator==(const ObjectType& other) const {

    CardType other_card = dynamic_cast<const CardType&>(other);
    return this->get_rank() == other_card.get_rank() && this->get_suit() == other_card.get_suit();
}

std::string CardType::to_string() const
{
    return this->get_id();
}

CardType Yolo_index_codec::yolo_index_to_card(int index){
    if (index < 0 || index >= 52) {
        CardType card(CardType::Rank::UNKNOWN, CardType::Suit::UNKNOWN);
        return card;
    }
    int r = index / 4;
    int s = index % 4;
    return CardType(static_cast<CardType::Rank>(r), static_cast<CardType::Suit>(s));
}

int Yolo_index_codec::card_to_yolo_index(const CardType& card){
    if (card.get_rank() == CardType::Rank::UNKNOWN || card.get_suit() == CardType::Suit::UNKNOWN) return noCardIndex;
    int index = static_cast<int>(card.get_rank()) * numSuits + static_cast<int>(card.get_suit());
    return index;
}


#include "../include/CardType.h"

#include "../include/Label.h"
#include "../include/Utils.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>


   

Card_Type::Card_Type(const std::string& card_text) {
    set_type(card_text);
}

void Card_Type::set_type(const std::string& card_text) {

    if (card_text.size() < 2) {
        rank = Rank::NOTHING; 
        suit = Suit::NOTHING; 
        return;
    }
    std::string str_norm = Utils::String::normalize(card_text);
    
    // suit = last char
    const char suit_ch = str_norm.back();
    str_norm.pop_back();  
    
    // rank = what remains
    const Rank& rk  = Card_Type::string_to_rank(str_norm);                 
    const Suit& su = Card_Type::string_to_suit(std::string(1, suit_ch)); 

    rank = rk;
    suit = su;
}


std::map<std::string, Card_Type::Rank> Card_Type::map_string_to_rank = {
    {"K",  Card_Type::Rank::K},
    {"Q",  Card_Type::Rank::Q},
    {"J",  Card_Type::Rank::J},
    {"10", Card_Type::Rank::R10},
    {"9",  Card_Type::Rank::R9},
    {"8",  Card_Type::Rank::R8},
    {"7",  Card_Type::Rank::R7},
    {"6",  Card_Type::Rank::R6},
    {"5",  Card_Type::Rank::R5},
    {"4",  Card_Type::Rank::R4},
    {"3",  Card_Type::Rank::R3},
    {"2",  Card_Type::Rank::R2},
    {"A",  Card_Type::Rank::A},
    {"UNKNOWN", Card_Type::Rank::UNKNOWN}
};

std::map<std::string, Card_Type::Suit> Card_Type::map_string_to_suit = {
    {"C", Card_Type::Suit::CLUBS},
    {"D", Card_Type::Suit::DIAMONDS},
    {"H", Card_Type::Suit::HEARTS},
    {"S", Card_Type::Suit::SPADES},
    {"UNKNOWN", Card_Type::Suit::UNKNOWN}
};

std::map<Card_Type::Rank, std::string> Card_Type::map_rank_to_string = Utils::Map::createInverseMap(Card_Type::map_string_to_rank);
std::map<Card_Type::Suit, std::string> Card_Type::map_suit_to_string = Utils::Map::createInverseMap(Card_Type::map_string_to_suit);



std::string Card_Type::to_string() const {

    const std::string r = map_rank_to_string[this->rank];
    const std::string s = map_suit_to_string[this->suit];
    
    return r + s; // ex: "10H", 10 of Hearts
}

const Card_Type::Rank Card_Type::string_to_rank(const std::string& s) {
    auto it = map_string_to_rank.find(s);
    return it != map_string_to_rank.end() ? it->second : Card_Type::Rank::UNKNOWN;
}

const Card_Type::Suit Card_Type::string_to_suit(const std::string& s) {
    auto it = map_string_to_suit.find(s);
    return it != map_string_to_suit.end() ? it->second : Card_Type::Suit::UNKNOWN;
}


bool operator<(const Card_Type& l, const Card_Type& r) {
    if (l.get_suit() != r.get_suit())
        return l.get_suit() < r.get_suit();
    return l.get_rank() < r.get_rank();
}

std::ostream& operator<<(std::ostream& os, const Card_Type& c) {
    return os << c.to_string();
}


Card_Type Yolo_index_codec::yolo_index_to_card(int index){
    if (index < 0 || index >= 52) {
        Card_Type card(Card_Type::Rank::UNKNOWN, Card_Type::Suit::UNKNOWN);
        return card;
    }
    int r = index / 4;
    int s = index % 4;
    return Card_Type(static_cast<Card_Type::Rank>(r), static_cast<Card_Type::Suit>(s));
}

int Yolo_index_codec::card_to_yolo_index(const Card_Type& card){
    if (card.get_rank() == Card_Type::Rank::UNKNOWN || card.get_suit() == Card_Type::Suit::UNKNOWN) return noCardIndex;
    int index = static_cast<int>(card.get_rank()) * numSuits + static_cast<int>(card.get_suit());
    return index;
}



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

std::map<CardType::Rank, std::string> CardType::map_rank_to_string = Utils::Map::createInverseMap(CardType::map_string_to_rank);
std::map<CardType::Suit, std::string> CardType::map_suit_to_string = Utils::Map::createInverseMap(CardType::map_string_to_suit);

std::string Card_Type::get_id() const {

    const std::string r = map_rank_to_string[this->rank];
    const std::string s = map_suit_to_string[this->suit];
    
    return r + s; // for example: "10H", 10 of Hearts
}

int Card_Type::get_id_number() const{
    return Yolo_index_codec::card_to_yolo_index(this->get_id());
}

const CardType::Rank CardType::string_to_rank(const std::string& s) {
    auto it = map_string_to_rank.find(s);
    return it != map_string_to_rank.end() ? it->second : Card_Type::Rank::UNKNOWN;
}

const CardType::Suit CardType::string_to_suit(const std::string& s) {
    auto it = map_string_to_suit.find(s);
    return it != map_string_to_suit.end() ? it->second : Card_Type::Suit::UNKNOWN;
}


bool Card_Type::operator<(const ObjectType& other) const {

    Card_Type other_card = dynamic_cast<const Card_Type&>(other);
    
    if (this->get_suit() != other_card.get_suit())
        return this->get_suit() < other_card.get_suit();
    return this->get_rank() < other_card.get_rank();
}

bool Card_Type::operator==(const ObjectType& other) const {

    Card_Type other_card = dynamic_cast<const Card_Type&>(other);
    return this->get_rank() == other_card.get_rank() && this->get_suit() == other_card.get_suit();
}

std::ostream &operator<<(std::ostream &os, const Card_Type &card) {
    return os << card.get_id();
}

CardType Yolo_index_codec::yolo_index_to_card(int index){
    if (index < 0 || index >= 52) {
        Card_Type card(Card_Type::Rank::UNKNOWN, Card_Type::Suit::UNKNOWN);
        return card;
    }
    int r = index / 4;
    int s = index % 4;
    return CardType(static_cast<CardType::Rank>(r), static_cast<CardType::Suit>(s));
}

int Yolo_index_codec::card_to_yolo_index(const Card_Type& card){
    if (card.get_rank() == Card_Type::Rank::UNKNOWN || card.get_suit() == Card_Type::Suit::UNKNOWN) return noCardIndex;
    int index = static_cast<int>(card.get_rank()) * numSuits + static_cast<int>(card.get_suit());
    return index;
}


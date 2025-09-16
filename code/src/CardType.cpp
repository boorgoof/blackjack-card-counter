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
        rank = Rank::UNKNOWN; 
        suit = Suit::UNKNOWN; 
        return;
    }
    std::string t = Utils::String::normalize(card_text);
    
    // suit = last char
    const std::string suit_tok(1, t.back());
    auto su = string_to_suit(suit_tok);
    
    // rank = what remains
    t.pop_back();
    auto rk = string_to_rank(t);
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
    {"A",  Card_Type::Rank::A}
};

std::map<std::string, Card_Type::Suit> Card_Type::map_string_to_suit = {
    {"C", Card_Type::Suit::CLUBS},
    {"D", Card_Type::Suit::DIAMONDS},
    {"H", Card_Type::Suit::HEARTS},
    {"S", Card_Type::Suit::SPADES}
};

std::map<Card_Type::Rank, std::string> Card_Type::map_rank_to_string = Utils::Map::createInverseMap(Card_Type::map_string_to_rank);
std::map<Card_Type::Suit, std::string> Card_Type::map_suit_to_string = Utils::Map::createInverseMap(Card_Type::map_string_to_suit);



std::string Card_Type::to_string() const {

    const std::string r = map_rank_to_string[this->rank];
    const std::string s = map_suit_to_string[this->suit];
    
    return r + s; // ex: "10H", 10 of Hearts
}


const Card_Type::Rank& Card_Type::string_to_rank(const std::string& s) {
    auto it = map_string_to_rank.find(s);
    return it != map_string_to_rank.end() ? it->second : Card_Type::Rank::UNKNOWN;
}

const Card_Type::Suit& Card_Type::string_to_suit(const std::string& s) {
    auto it = map_string_to_suit.find(s);
    return it != map_string_to_suit.end() ? it->second : Card_Type::Suit::UNKNOWN;
}


// ordering: prima suit poi rank
bool operator<(const Card_Type& l, const Card_Type& r) {
    if (l.get_suit() != r.get_suit())
        return l.get_suit() < r.get_suit();
    return l.get_rank() < r.get_rank();
}

std::ostream& operator<<(std::ostream& os, const Card_Type& c) {
    return os << c.to_string();
}




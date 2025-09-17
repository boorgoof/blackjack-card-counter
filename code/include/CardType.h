#ifndef CARD_TYPE_H
#define CARD_TYPE_H

#include <string>
#include <map>


class Card_Type {

public:
    
    enum class Suit {
        CLUBS,      // C
        DIAMONDS,   // D
        HEARTS,     // H
        SPADES,     // S
        UNKNOWN
    };

    enum class Rank {
        R2, R3, R4, R5, R6,
        R7, R8, R9,
        R10, J, Q, K, A,
        UNKNOWN
    };

    
    Card_Type(const std::string& card_text);
    Card_Type(const Rank& r, const Suit& s) : rank{r}, suit{s} {}
    
    const Rank& get_rank() const { return this->rank; }
    const Suit& get_suit() const { return this->suit; }

    void set_rank(const Rank& r) { this->rank = r; }
    void set_suit(const Suit& s) { this->suit = s; }
    void set_type(const std::string& card_text);

    std::string to_string() const;

    static const Rank string_to_rank(const std::string& r);
    static const Suit string_to_suit(const std::string& s);

    
    static std::map<std::string, Rank> map_string_to_rank;
    static std::map<std::string, Suit> map_string_to_suit;

    static std::map<Rank, std::string> map_rank_to_string;
    static std::map<Suit, std::string> map_suit_to_string;

private:
   
    Rank rank{Rank::UNKNOWN};
    Suit suit{Suit::UNKNOWN};
};

std::ostream& operator<<(std::ostream& os, const Card_Type& c); 

namespace blackjack {

  enum class HiLo { Pos=+1, Neutral=0, Neg=-1, Unknown=9999 };
  inline int HiLo_to_int(HiLo v) { return v==HiLo::Pos ? 1 : (v==HiLo::Neg ? -1 : 0); }
  
  inline HiLo rank_to_HiLo(Card_Type::Rank r) {
    
    using R = Card_Type::Rank;

    switch (r) {
      case R::R2: case R::R3: case R::R4: case R::R5: case R::R6: return HiLo::Pos;
      case R::R7: case R::R8: case R::R9:                         return HiLo::Neutral;
      case R::R10: case R::J: case R::Q: case R::K: case R::A:    return HiLo::Neg;
      default:                                                    return HiLo::Unknown;
    }
  }

  
}

#endif // CARD_TYPE_H
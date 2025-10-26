#ifndef CARD_TYPE_H
#define CARD_TYPE_H

#include <string>
#include <map>
#include "ObjectType.h"

class Card_Type : public ObjectType {

public:

  enum class Suit {
      SPADES = 0,     // S
      CLUBS = 1,    // C
      DIAMONDS = 2,   // D
      HEARTS = 3,     // H
      UNKNOWN = -1
  };

  enum class Rank {
      R2 = 1, R3 = 2, R4 = 3, R5 = 4, R6 = 5,
      R7 = 6, R8 = 7, R9 = 8,
      R10 = 9, J = 10, Q = 11, K = 12, A = 0,
      UNKNOWN = -1
  };

  
  Card_Type(const std::string& card_text);
  Card_Type(const Rank& r, const Suit& s) : rank{r}, suit{s} {}

  Card_Type(const Card_Type& other) : rank{other.rank}, suit{other.suit} {}

  Card_Type& operator=(const Card_Type& other) {
      this->rank = other.rank;
      this->suit = other.suit;
      return *this;
  }

  std::unique_ptr<ObjectType> clone() const { return std::make_unique<Card_Type>(*this); }

  std::string get_id() const;
  int get_id_number() const;
  bool isValid() const { return this->rank != Rank::UNKNOWN && this->suit != Suit::UNKNOWN; } 
  
  const Rank& get_rank() const { return this->rank; }
  const Suit& get_suit() const { return this->suit; }
  const std::string get_type() const { return this->get_id(); }
  
  void set_rank(const Rank& r) { this->rank = r; }
  void set_suit(const Suit& s) { this->suit = s; }
  void set_type(const std::string& card_text);

  static const Rank string_to_rank(const std::string& r);
  static const Suit string_to_suit(const std::string& s);

  static std::map<std::string, Rank> map_string_to_rank;
  static std::map<std::string, Suit> map_string_to_suit;

  static std::map<Rank, std::string> map_rank_to_string;
  static std::map<Suit, std::string> map_suit_to_string;

  bool operator<(const ObjectType& other) const;
  bool operator==(const ObjectType& other) const;

private:
   
  Rank rank{Rank::UNKNOWN};
  Suit suit{Suit::UNKNOWN};
};

std::ostream& operator<<(std::ostream& os, const Card_Type& card); 


namespace Yolo_index_codec {

  constexpr int numRanks = 13;  
  constexpr int numSuits = 4;   
  constexpr int noCardIndex = numRanks * numSuits;  

  Card_Type yolo_index_to_card(int index);
  int card_to_yolo_index(const Card_Type& card); 

}


namespace Blackjack {

  enum class HiLo { Pos=+1, Neutral=0, Neg=-1 };
  inline int HiLo_to_int(HiLo v) { return v==HiLo::Pos ? 1 : (v==HiLo::Neg ? -1 : 0); }
  
  inline HiLo rank_to_HiLo(Card_Type::Rank r) {
    
    using R = Card_Type::Rank;

    switch (r) {
      case R::R2: case R::R3: case R::R4: case R::R5: case R::R6: return HiLo::Pos;
      case R::R7: case R::R8: case R::R9: return HiLo::Neutral;
      case R::R10: case R::J: case R::Q: case R::K: case R::A: return HiLo::Neg;
      default: return HiLo::Neutral;
    }
  }

  
}

#endif // CARD_TYPE_H
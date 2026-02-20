// Federico Meneghetti

#ifndef CARD_TYPE_H
#define CARD_TYPE_H

#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include "ObjectType.h"

class CardType : public ObjectType {

public:

  /**
   * @brief enum to represent the suit of a card. 
   */
  enum class Suit {
    SPADES = 0,     // S
    CLUBS = 1,    // C
    DIAMONDS = 2,   // D
    HEARTS = 3,     // H
    UNKNOWN = -1
  };

  /**
   * @brief enum to represent the rank of a card.
   */
  enum class Rank {
    
    A = 0, 
    R2 = 1, 
    R3 = 2,
    R4 = 3, 
    R5 = 4, 
    R6 = 5,
    R7 = 6, 
    R8 = 7, 
    R9 = 8,
    R10 = 9, 
    J = 10, 
    Q = 11, 
    K = 12, 
    UNKNOWN = -1
  };

 

  
  CardType(const std::string& card_text);
  CardType(const Rank& r, const Suit& s) : rank{r}, suit{s} {}

  CardType(const CardType& other) : rank{other.rank}, suit{other.suit} {}

  CardType& operator=(const CardType& other) {
      this->rank = other.rank;
      this->suit = other.suit;
      return *this;
  }

  std::unique_ptr<ObjectType> clone() const { return std::make_unique<CardType>(*this); }

  std::string get_id() const;
  int get_id_number() const;
  bool isValid() const { return this->rank != Rank::UNKNOWN && this->suit != Suit::UNKNOWN; } 
  
  //getters
  const Rank& get_rank() const { return this->rank; }
  const Suit& get_suit() const { return this->suit; }
  const std::string get_type() const { return this->get_id(); }
  // setters
  void set_rank(const Rank& r) { this->rank = r; }
  void set_suit(const Suit& s) { this->suit = s; }
  void set_type(const std::string& card_text);

  // to string
  static const Rank string_to_rank(const std::string& r);
  static const Suit string_to_suit(const std::string& s);

  /**
   * @brief maps the string representation of a rank to the corresponding Rank enum value.
   */
  static std::map<std::string, Rank> map_string_to_rank;

  /**
   * @brief maps the string representation of a suit to the corresponding Suit enum value.
   */
  static std::map<std::string, Suit> map_string_to_suit;

  /**
   * @brief maps the Rank enum value to its corresponding string representation.
   */
  static std::map<Rank, std::string> map_rank_to_string;

  /**
   * @brief maps the Suit enum value to its corresponding string representation.
   */
  static std::map<Suit, std::string> map_suit_to_string;

  

  bool operator<(const ObjectType& other) const;
  bool operator==(const ObjectType& other) const;

  std::string to_string() const;

private:
   
  Rank rank{Rank::UNKNOWN};
  Suit suit{Suit::UNKNOWN};
};

namespace card_color_utils { 

  /**
   * @brief enum to represent the color of a card.
   */
  enum class CardColor {
    RED,
    BLACK,
    UNKNOWN
  };

  CardColor suit_to_color(CardType::Suit suit);
  std::vector<CardType::Suit> color_to_suits(CardColor color);
  cv::Scalar to_scalar(CardColor c);

}

/**
 * @brief namespace to encode and decode the card types using YOLO index conventions.
 */
namespace Yolo_index_codec {

  constexpr int numRanks = 13;  
  constexpr int numSuits = 4;   
  constexpr int noCardIndex = numRanks * numSuits;  

  /**
   * @brief maps the index of a detected class from the YOLO model to a CardType object.
   */
  CardType yolo_index_to_card(int index);

  /**
   * @brief maps a CardType object to the corresponding index used in the YOLO model for detection.
   */
  int card_to_yolo_index(const CardType& card); 

}


namespace Blackjack {

  enum class HiLo { Pos=+1, Neutral=0, Neg=-1 };
  inline int HiLo_to_int(HiLo v) { return v==HiLo::Pos ? 1 : (v==HiLo::Neg ? -1 : 0); }
  
  /**
   * @brief maps the rank of a card to its corresponding Hi-Lo count value.
   */
  inline HiLo rank_to_HiLo(CardType::Rank r) {
    
    using R = CardType::Rank;

    switch (r) {
      case R::R2: case R::R3: case R::R4: case R::R5: case R::R6: return HiLo::Pos;
      case R::R7: case R::R8: case R::R9: return HiLo::Neutral;
      case R::R10: case R::J: case R::Q: case R::K: case R::A: return HiLo::Neg;
      default: return HiLo::Neutral;
    }
  }


  /**
   * @brief maps a Hi-Lo count value to a corresponding OpenCV color for visualization.
   */
  inline cv::Scalar HiLo_to_cv_color(HiLo v) {
    
    switch (v) {
      case HiLo::Pos: return cv::Scalar(0, 255, 0);  // Green
      case HiLo::Neutral: return cv::Scalar(255, 0, 0);  // Blue
      case HiLo::Neg: return cv::Scalar(0, 0, 255);  // Red
      default: return cv::Scalar(255, 0, 0);   
    }
  }

  
}

#endif // CARD_TYPE_H
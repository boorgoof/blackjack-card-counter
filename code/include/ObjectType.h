#ifndef OBJECT_TYPE_H
#define OBJECT_TYPE_H
#include <string>
#include <memory>
#include <ostream>

class ObjectType {

public:

    virtual ~ObjectType() = 0;

    ObjectType() = default;

    ObjectType(const ObjectType&) = delete;
    ObjectType& operator=(const ObjectType&) = delete;

    virtual std::unique_ptr<ObjectType> clone() const = 0;
    virtual std::string get_id() const = 0;
    virtual int get_id_number() const = 0;
    virtual bool isValid() const = 0;

    virtual bool operator<(const ObjectType& other) const = 0;
    virtual bool operator==(const ObjectType& other) const  = 0;

    virtual std::string to_string() const = 0;

};

inline std::ostream& operator<<(std::ostream& os, const ObjectType& obj) {
    os << obj.to_string();
    return os;
}

#endif // OBJECT_TYPE_H
#ifndef OBJECT_TYPE_H
#define OBJECT_TYPE_H
#include <string>
#include <memory>

class ObjectType {

public:

    virtual ~ObjectType() = 0;

    ObjectType() = default;

    ObjectType(const ObjectType&) = delete;
    ObjectType& operator=(const ObjectType&) = delete;

    virtual std::unique_ptr<ObjectType> clone() const = 0;
    virtual std::string get_id() const = 0;
    virtual bool isValid() const = 0;

    virtual bool operator<(const ObjectType& other) const = 0;
    virtual bool operator==(const ObjectType& other) const  = 0;

};



#endif // OBJECT_TYPE_H
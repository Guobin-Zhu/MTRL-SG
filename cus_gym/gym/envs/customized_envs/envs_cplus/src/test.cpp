#include <iostream>
#include <typeinfo>

int main() {
    int i = 10;
    float f = 3.14;
    double d = 2.718;
    char c = 'A';
    bool b = true;

    std::cout << "Type of i: " << typeid(i).name() << std::endl;
    std::cout << "Type of f: " << typeid(f).name() << std::endl;
    std::cout << "Type of d: " << typeid(d).name() << std::endl;
    std::cout << "Type of c: " << typeid(c).name() << std::endl;
    std::cout << "Type of b: " << typeid(b).name() << std::endl;

    return 0;
}




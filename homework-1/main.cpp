#include <iostream>
#include <chrono>
#include <cmath>
#include "matrix.hpp"

using namespace std;
using namespace std::chrono;

//Compile command:
//g++ -c src/matrix.cpp -o main -I./include -O3 -std=c++11 

int main(void) {

    int m = 100;
    int n = 100;
    Matrix A = Matrix(m, n);

    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            A(i,j) = 1.0;
        }
    }

    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            std::cout << A(i,j) << std::endl;
        }
    }






}


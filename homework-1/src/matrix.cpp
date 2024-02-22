#include <iostream>
#include "matrix.hpp"

using std::logic_error;

//Overloaded operator to return matrix value at row i, column j
double & Matrix::operator()(int row, int column){

    //Bounds checking
    if (row > num_rows() || row < 0){
        //TODO:
            //Check that this is what bounds error is called
        throw logic_error("Row index is outside of matrix bounds");
    
    } else if (column > num_columns() || column < 0){
        //TODO:
            //Check that this is what bounds error is called
        throw logic_error("Column index is outside of matrix bounds");
    }
    
    return data[row * num_columns() + column];

}
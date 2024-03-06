#include <iostream>
#include "matrix.hpp"

using namespace std;
using std::logic_error;

//Overloaded operator to return matrix value at row i, column j
double & Matrix::operator()(int row, int column){

    //Bounds checking
    if (row >= num_rows() || row < 0){
        cout << row << " is >= " << num_rows() << endl;
        //TODO:
            //Check that this is what bounds error is called
        throw logic_error("Row index is outside of matrix bounds");
    
    } else if (column >= num_columns() || column < 0){
        cout << column << " is >= " << num_columns() << endl;
        //TODO:
            //Check that this is what bounds error is called
        throw logic_error("Column index is outside of matrix bounds");
    }
    
    return _data[row * num_columns() + column];

}
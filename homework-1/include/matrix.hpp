#ifndef _MATRIX
#define _MATRIX

class Matrix{

public:

    //Constructor of Matrix Class
    Matrix(int rows, int columns){
        //Number of rows in matrix
        rows = rows;
        
        //Number of columns in marix
        columns = columns;

        //Data contained in matrix
        data = new double [rows * columns];
    }

    //Destructor for Matrix Class
    ~Matrix(){
        delete [] data;
    }

    //Returns number of rows of matrix
    int num_rows() const { return rows; };
    
    //Returns number of columns of matrix
    int num_columns() const {return columns; };

    //Overloaded operator to return matrix value at row i, column j
    double & operator()(int row, int column);

private:

    double * data;
    int rows;
    int columns;

};

#endif
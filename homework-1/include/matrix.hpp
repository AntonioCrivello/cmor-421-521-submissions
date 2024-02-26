#ifndef _MATRIX
#define _MATRIX

class Matrix{

public:

    //Constructor of Matrix Class
    Matrix(int rows, int columns){
        //Number of rows in matrix
        _rows = rows;
        
        //Number of columns in marix
        _columns = columns;

        //Data contained in matrix
        _data = new double [rows * columns];
    }

    //Copy Constructor
    Matrix(const Matrix & copyMatrix){
        _rows = copyMatrix.num_rows();
        _columns = copyMatrix.num_columns();
        _data = new double [_rows * _columns];
        for (int i = 0; i < _rows; ++i){
            for (int j = 0; j < _columns; ++j){
                _data[i * num_columns() + j] = copyMatrix._data[i * num_columns() + j];
            }
        }
    }

    //Returns number of rows of matrix
    int num_rows() const { return _rows; };
    
    //Returns number of columns of matrix
    int num_columns() const {return _columns; };

    //Destructor for Matrix Class
    ~Matrix(){
        delete [] _data;
    }

    //Overloaded operator to return matrix value at row i, column j
    double & operator()(int row, int column);

private:

    double * _data;
    int _rows;
    int _columns;

};

#endif
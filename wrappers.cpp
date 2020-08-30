#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ostream>
#include <iostream>
#include <list>

using namespace std;

int add(int i, int j) {
    return i + j;
}
int sub(int i , int j){
    return i - j;
}

void printArr(list<int> l){
    cout<<"Printing array"<<endl;
    for(int n : l){
        cout<<n<<endl;
    }

}


PYBIND11_MODULE(kf_cpp, m) {
    m.doc() = "C++ KF implementation wrappers"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");
    m.def("sub", &sub, "A function which subtracts two numbers");
    m.def("printArr", &printArr, "A function which prints out an array");//does this work with a numpy array?
}
#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int main(int, char**)
{

	mat X;
	X.load("Iris.csv");
	
	X.shed_col(X.n_cols-1);

	cout<<X<<endl;

	return 0;
}

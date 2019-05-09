// https://hackernoon.com/introduction-to-machine-learning-algorithms-logistic-regression-cbdd82d81a36

// g++ lr.cpp test.cpp -o lr -O2 -larmadillo

// ./lr to run

#include "lr.hpp"

using namespace std;
using namespace arma;

int main(int, char**)
{

	mat X;
	X.load("Iris.csv");
	int i = X.n_cols-1;
	vec y = X.col(i);
	X.shed_col(i);
	//cout<<y<<endl;

	LR lr;

	lr.train(X,y);
	vec w = lr.getW();

	//cout<<w<<endl;

	LR lr2;

	lr2.setW(w);

	vec y2 = lr2.predict(X);
	
	cout<<y2<<endl;
	
	return 0;
}

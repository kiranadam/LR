#ifndef LR_HPP
#define LR_HPP

#include <iostream>
#include <armadillo>
#include <cmath>

using namespace std;
using namespace arma;

class LR
{

	private:
		double alpha;     // learning rate
		int epoch;        // maximum iterations
		double lambda;    // L2 regularization
		double tolerance; // error tolerance
		rowvec w;         // weights 		
		
		
		double sigmoid(double x); // Defination for logistic function

		double cost(vec &y, vec &h); // Defination for cost function

	public:
		LR();

		LR(int epoch, double alpha, double lambda, double tolerance);

		~LR();
};


#endif

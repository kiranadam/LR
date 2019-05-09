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
		double tolerance; // error tolerance
		vec W;            // weights 		
		
		double sigmoid(double x); // Defination for logistic function

		
	public:
		LR(); // default constructor

		LR(int epoch, double alpha,double tolerance);  // parameterized constructor

		~LR();

		double cost(vec y, vec h); // Defination for cost function

		void train(mat X, vec y);  // defination to train a function

		vec predict_prob(mat X);   //

		vec predict(mat X);

		vec getW();               // get Weights
		
		void setW(vec W);	  // set weights

};


#endif

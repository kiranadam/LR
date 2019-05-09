// g++ lr.cpp -o lr -O2 -larmadillo

#include "lr.hpp"

//Default Constructor 
LR::LR()
{
	alpha = 0.01;
	epoch = 200;
	lambda = 0.05;
	tolerance = 0.01;
}

// Parameterized Constructor
LR::LR(int epoch, double alpha, double lambda, double tolerance)
{
	this->epoch = epoch;
	this->alpha = alpha;
	this->lambda = lambda;
	this->torlerance = tolerance;
}

//Logistic function
double LR::sigmoid(double x)
{
	return 1.0/(1.0 + exp(-x));
}

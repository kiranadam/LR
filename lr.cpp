// g++ lr.cpp -o lr -O2 -larmadillo

#include "lr.hpp"
#include <cmath>

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

// Destructor
LR::~LR()
{
}

//Logistic function
double LR::sigmoid(double x)
{
	return 1.0/(1.0 + exp(-x));
}

// Cost function
double LR::double cost(vec y, vec h)
{
	int m = y.size();  // get the vector size
	double c;
	
	for(size_t i=0; i<m; ++i)
	{
		c -= (y(i)*log2(h(i))+((1.0-y(i))*log2(1.0-h(i)));
	}
	return c/m;
}
		      

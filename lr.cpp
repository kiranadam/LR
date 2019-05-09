
#include "lr.hpp"
#include <cmath>


//Default Constructor 
LR::LR()
{
	alpha = 0.0001;
	epoch = 10000;
	tolerance = 0.1;
}

// Parameterized Constructor
LR::LR(int epoch, double alpha, double tolerance)
{
	this->epoch = epoch;
	this->alpha = alpha;
	this->tolerance = tolerance;
}

// Destructor
LR::~LR()
{
}

//Logistic function
double LR::sigmoid(double x)
{
	return 1.0/(1.0 + std::exp(-x));
}


// Cost function
double LR::cost(vec y, vec h)
{
	int m = y.size();  // get the vector size
	double c;
	
	for(size_t i=0; i<m; i++)
	{
		
		c -= (y(i)*std::log2(h(i)))+((1.0-y(i))*std::log2(1.0-h(i))); // calculate cost

		//cout<<"y = "<<y<<"  h = "<<h<<"at i ="<<i<<"  cost = "<<c<<endl; 
	}

	return c/m;
}

// training function for dataset
void LR::train(mat X, vec y)
{
	W = ones<vec>(X.n_cols+1); // weights initialization

	mat new_X = join_horiz(X, ones<mat>(X.n_rows,1)); // adding last columns of 1's

	for(size_t i=0; i<epoch; i++)
	{
		vec pred_y = predict_prob(X);

		W = W - alpha*new_X.t()*(pred_y-y);
		
		double c = cost(y,predict_prob(X));
		
		if(c<=tolerance) break; 
	}
}


// calcuating the expected output	     
vec LR::predict_prob(mat X)
{
	//predict the probability (of label 1) for given data X
	mat new_X = join_horiz(X, ones<vec>(X.n_rows)); // adding last columns of 1's

	int m = new_X.n_rows;

	vec y_pred_prob(m);

	for(size_t i=0; i<m; i++)
	{
		mat z = new_X.row(i)*W;
		
		y_pred_prob(i) = sigmoid(z(0,0));
		
	}

	return y_pred_prob;
}


// predict the output
vec LR::predict(mat X)
{
	//predict the label for given data X
	 
	vec y_pred_prob = predict_prob(X);
	int m = y_pred_prob.size();
	vec y_pred(m);

	for(size_t i=0; i<m; i++)
	{
		y_pred(i) = y_pred_prob(i)>=0.5?1:0;
	}
	return y_pred;
}

// get weights 
vec LR::getW()
{
	return W;
}

// set weights just for test purpose
void LR::setW(vec W)
{
	this->W = W;
}

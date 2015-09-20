//Sully Chen 2015

#pragma once
#ifndef FFANN_H
#define FFANN_H

#include <cmath>
#include <vector>
#include "MatrixMath.h"

class FFANN
{
public:
	FFANN(int* dimensions, int num_layers);
	std::vector<Matrix> Weights;
	std::vector<Matrix> Biases;
	int* Dimensions;
	int Num_Layers;
	std::vector<Matrix> FeedForward(Matrix input);
	float TrainWithBackPropagation(Matrix input, Matrix output, float learning_rate);
};

FFANN BreedNetworks(FFANN Parent1, FFANN Parent2, float mutation_probability, float mutation_range);

#endif
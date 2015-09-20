//Sully Chen 2015

#include <cmath>
#include <vector>
#include "MatrixMath.h"
#include "FFANN.h"

FFANN::FFANN(int* dimensions, int num_layers)
{
	Dimensions = dimensions;
	Num_Layers = num_layers;

	//the first elements of the weights matrix vector object is just a filler to make the math look cleaner, it's not actually used
	Matrix temp;
	Weights.push_back(temp);

	//create randomized weight matrices
	for (int i = 0; i < num_layers - 1; i++)
	{
		Matrix m(dimensions[i], dimensions[i + 1]);
		for (int j = 0; j < dimensions[i]; j++)
		{
			for (int k = 0; k < dimensions[i + 1]; k++)
			{
				m.Elements[j * m.Dimensions[1] + k] = (rand() % 200 - 100) / 1000.0f;
			}
		}
		Weights.push_back(m);
	}

	//create biases
	for (int i = 0; i < num_layers; i++)
	{
		Matrix m(dimensions[i], 1);
		for (int j = 0; j < dimensions[i]; j++)
		{
			m.Elements[j] = (rand() % 200 - 100) / 1000.0f;
		}
		Biases.push_back(m);
	}
}

std::vector<Matrix> FFANN::FeedForward(Matrix input)
{
	std::vector<Matrix> outputs;
	//Add biases and apply activation function to each input element
	for (int i = 0; i < input.Dimensions[0]; i++)
	{
		input.Elements[i] += Biases[0].Elements[i];
		input.Elements[i] = 1 / (1 + pow(2.718281828459f, -input.Elements[i]));
	}
	outputs.push_back(input);

	//feed forward calculation
	for (int i = 1; i < Num_Layers; i++)
	{
		//feed forward
		Matrix z;
		z = Weights[i].Transpose() * outputs[i - 1] + Biases[i];

		outputs.push_back(z);

		//Apply activation function
		for (int j = 0; j < outputs[i].Dimensions[0]; j++)
		{
			outputs[i].Elements[j] = 1 / (1 + pow(2.718281828459f, -outputs[i].Elements[j]));
		}
	}

	return outputs;
}

float FFANN::TrainWithBackPropagation(Matrix input, Matrix output, float learning_rate)
{
	std::vector<Matrix> outputs = FeedForward(input);

	std::vector<Matrix> temp_deltas; //layer deltas stored backwards in order

	//calculate cost function
	float cost = 0.0f;
	Matrix partial_cost_matrix(Dimensions[Num_Layers - 1], 1);
	partial_cost_matrix = output + (outputs[outputs.size() - 1] * -1);
	for (int i = 0; i < partial_cost_matrix.Elements.size(); i++)
	{
		cost += 0.5f * partial_cost_matrix.Elements[i] * partial_cost_matrix.Elements[i];
	}
	//calculate last layer deltas
	Matrix lld(Dimensions[Num_Layers - 1], 1);
	lld = outputs[outputs.size() - 1] + (output * -1);
	for (int i = 0; i < lld.Dimensions[0]; i++)
	{
		float a = outputs[outputs.size() - 1].Elements[i];
		lld.Elements[i] *= a * (1 - a); //derivative of activation function
	}
	temp_deltas.push_back(lld);

	//calculate the rest of the deltas through back propagation
	int j = 0; //this keeps track of the index for the next layer's delta
	for (int i = Num_Layers - 2; i >= 0; i--) //start at the second to last layer
	{
		Matrix delta(Dimensions[i], 1);
		delta = Weights[i + 1] * temp_deltas[j];
		j++;
		for (int k = 0; k < delta.Dimensions[0]; k++)
		{
			float a = outputs[i].Elements[k];
			delta.Elements[k] *= a * (1 - a); //derivative of activation function
		}
		temp_deltas.push_back(delta);
	}

	//put the deltas into a new vector object in the correct order
	std::vector<Matrix> deltas;
	for (int i = temp_deltas.size() - 1; i >= 0; i--)
	{
		deltas.push_back(temp_deltas[i]);
	}

	//update biases
	for (int i = 0; i < Biases.size(); i++)
	{
		Biases[i] = Biases[i] + deltas[i] * (-1.0f * learning_rate);
	}

	//update weights
	for (int i = 1; i < Weights.size(); i++)
	{
		Weights[i] = Weights[i] + ((outputs[i - 1] * deltas[i].Transpose()) * (-1.0f * learning_rate));
	}

	return cost;
}

FFANN BreedNetworks(FFANN Parent1, FFANN Parent2, float mutation_probability, float mutation_range)
{
	if (mutation_probability > 1.0f)
	{
		mutation_probability = 1.0f;
		std::cout << "Warning: keep mutation probability between 0.0 and 1.0. Capping to 1.0" << std::endl;
	}
	else if (mutation_probability < 0.0f)
	{
		mutation_probability = 0.0f;
		std::cout << "Warning: keep mutation probability between 0.0 and 1.0. Flooring to 0.0" << std::endl;
	}
	//Make sure the networks are the same size
	if (Parent1.Num_Layers != Parent2.Num_Layers)
	{
		std::cout << "Error! Cannot breed due to network formating!" << std::endl;
		FFANN ffann(0, 0);
		return ffann;
	}
	for (int i = 0; i < Parent1.Num_Layers; i++)
	{
		if (Parent1.Dimensions[i] != Parent2.Dimensions[i])
		{
			std::cout << "Error! Cannot breed due to network formating!" << std::endl;
			FFANN ffann(0, 0);
			return ffann;
		}
	}

	//Genetic algorithm
	FFANN offspringnetwork(Parent1.Dimensions, Parent1.Num_Layers); //initialize offspring network

	//crossover the genes of the weights
	for (int i = 1; i < Parent1.Num_Layers; i++) //we start at 1 because weights[0] is a filler matrix and does not contain any elements
	{
		int crossoverpoint = rand() % Parent1.Weights[i].Elements.size();
		for (int j = 0; j < crossoverpoint; j++)
		{
			offspringnetwork.Weights[i].Elements[j] = Parent1.Weights[i].Elements[j]; //one part of the gene is from parent 1
		}
		for (int j = crossoverpoint; j < offspringnetwork.Weights[i].Elements.size(); j++)
		{
			offspringnetwork.Weights[i].Elements[j] = Parent2.Weights[i].Elements[j]; //the other part of the gene is from parent 2
		}

		//randomly mutate genes
		for (int k = 0; k < offspringnetwork.Weights[i].Elements.size(); k++)
		{
			int random_int = rand() % (int)((1.01f - mutation_probability) * 1000); //we round mutation_probability * 1000 to an integer to ensure it is not a float
			for (int j = 0; j < 10; j++) //we're choosing out of 1000 to get precision up to the hundredth place, so we must take 10 samples to get a probability out of 100
			{
				//random selection of gene
				if (random_int == rand() % (int)(mutation_probability * 1000))
				{
					offspringnetwork.Weights[i].Elements[k] += (rand() % (int)(mutation_range * 20000 - mutation_range * 10000)) / 10000.0f; //mutate the gene
				}
			}
		}
	}

	//crossover the genes of the biases
	for (int i = 0; i < Parent1.Num_Layers; i++)
	{
		int crossoverpoint = rand() % Parent1.Biases[i].Elements.size();
		for (int j = 0; j < crossoverpoint; j++)
		{
			offspringnetwork.Biases[i].Elements[j] = Parent1.Biases[i].Elements[j]; //one part of the gene is from parent 1
		}
		for (int j = crossoverpoint; j < offspringnetwork.Biases[i].Elements.size(); j++)
		{
			offspringnetwork.Biases[i].Elements[j] = Parent2.Biases[i].Elements[j]; //the other part of the gene is from parent 2
		}

		//randomly mutate genes
		for (int k = 0; k < offspringnetwork.Biases[i].Elements.size(); k++)
		{
			int random_int = rand() % (int)((1.01f - mutation_probability) * 1000); //we round mutation_probability * 1000 to an integer to ensure it is not a float
			for (int j = 0; j < 10; j++) //we're choosing out of 1000 to get precision up to the hundredth place, so we must take 10 samples to get a probability out of 100
			{
				//random selection of gene
				if (random_int == rand() % (int)(mutation_probability * 1000))
				{
					offspringnetwork.Biases[i].Elements[k] += (rand() % (int)(mutation_range * 20000 - mutation_range * 10000)) / 10000.0f; //mutate the gene
				}
			}
		}
	}

	return offspringnetwork;
}

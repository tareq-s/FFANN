//Sully Chen 2015
//This is example code which creates a neural network and trains it to recognize the larger number out of a pair of floats ranging -1 to 1
//I ran this code with 1000000 examples at a learning rate of 0.001, and achieved 95% to 99% accuracy. I recommend these training settings
#include <iostream>
#include <vector>
#include <cmath>
#include <time.h>
#include <fstream>
#include <MatrixMath.h>
#include <FFANN.h>

int main()
{
	//seed random number generator
	srand(time(NULL));
	//create structure of neural network: 2 input neurons, 2 output neurons
	int dimensions[2] = { 2, 2 };
	//create the neural network
	FFANN testFFANN(dimensions, 2);

	int num_examples;
	float learning_rate = 0.0f;

	std::cout << "Train how many trials?: ";
	std::cin >> num_examples;
	std::cout << "At what learning rate?: ";
	std::cin >> learning_rate;

	const float original_learning_rate = learning_rate;

	std::ofstream saveFile("cost_function_data.txt");

	//run training
	for (int i = 0; i < num_examples; i++)
	{
		//create inputs
		float inputs[2] = { (rand() % 200 - 100) / 100.0f, (rand() % 200 - 100) / 100.0f };
		if (inputs[0] == inputs[1])
			inputs[0] += (rand() % 200 - 100) / 1000.0f;

		//create a 2x1 matrix
		Matrix input(2, 1, inputs);

		//create outputs based on which input is larger
		float outputs[2];
		if (inputs[0] > inputs[1])
		{
			outputs[0] = 1.0f;
			outputs[1] = 0.0f;
		}
		else
		{
			outputs[0] = 0.0f;
			outputs[1] = 1.0f;
		}
		Matrix output(2, 1, outputs);

		//lower learning rate near the end of training
		if (i > num_examples * 0.8f)
			learning_rate = original_learning_rate * 0.1f;

		//save cost function data
		float cost = testFFANN.TrainWithBackPropagation(input, output, learning_rate);
		if (i % (int)(num_examples / 100.0f) == 0)
		{
			saveFile << i << ", " << cost;
		}
		if (i % (int)(num_examples / 100.0f) == 0)
		{
			saveFile << "\n";
		}
	}
	saveFile.close();
	std::cout << "Done training!\n" << std::endl << "Testing feed forward neural network...\n" << std::endl;

	//test the network
	int num_trials = 10000;
	int num_correct = 0;

	for (int i = 0; i < num_trials; i++)
	{
		float inputs[2] = { 0.0f, 0.0f };
		while (inputs[0] == inputs[1])
		{
			inputs[0] = (rand() % 200 - 100) / 100.0f;
			inputs[1] = (rand() % 200 - 100) / 100.0f;
		}
		Matrix input(2, 1, inputs);
		std::vector<Matrix> outputs = testFFANN.FeedForward(input);
		if (inputs[0] > inputs[1])
		{
			if (outputs[outputs.size() - 1].Elements[0] > outputs[outputs.size() - 1].Elements[1])
				num_correct++;
		}
		else
		{
			if (outputs[outputs.size() - 1].Elements[0] < outputs[outputs.size() - 1].Elements[1])
				num_correct++;
		}
	}

	std::cout << "Done Testing! Here are the results:\n" << std::endl << "Accuracy: " << num_correct * 100 / num_trials << "% correct" << std::endl;

	system("PAUSE");
	return 0;
}
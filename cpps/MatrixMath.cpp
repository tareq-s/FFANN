//Sully Chen 2015

#include "MatrixMath.h"
#include <iostream>
#include <vector>

//initialize empty matrix
Matrix::Matrix()
{
}

Matrix::~Matrix()
{
}

//initialize matrix full of zeroes, if dimensions are specified
Matrix::Matrix(int row_dim, int col_dim)
{
	Dimensions[0] = row_dim;
	Dimensions[1] = col_dim;
	for (int i = 0; i < row_dim * col_dim; i++)
	{
		Elements.push_back(0.0f);
	}
}

//initialize matrix with elements specified
Matrix::Matrix(int row_dim, int col_dim, float* elements)
{
	Dimensions[0] = row_dim;
	Dimensions[1] = col_dim;
	for (int i = 0; i < row_dim * col_dim; i++)
	{
		Elements.push_back(0.0f);
	}
	for (int i = 0; i < row_dim * col_dim; i++)
	{
		Elements[i] = elements[i];
	}
}

//matrix multiplication
Matrix Matrix::operator*(Matrix& m)
{
	Matrix out(this->Dimensions[0], m.Dimensions[1]);
	if (this->Dimensions[1] != m.Dimensions[0])
	{
		std::cout << "Error: Invalid matrix multiplication operation " <<
			this->Dimensions[0] << "x" << this->Dimensions[1] << " * " <<
			m.Dimensions[0] << "x" << m.Dimensions[1] << std::endl;
		return out;
	}

	for (int i = 0; i < this->Dimensions[0]; i++)
	{
		for (int j = 0; j < m.Dimensions[1]; j++)
		{
			float sum = 0.0f;
			for (int k = 0; k < this->Dimensions[1]; k++)
			{
				sum += this->Elements[i * this->Dimensions[1] + k]
					* m.Elements[k * m.Dimensions[1] + j];
			}
			out.Elements[i * out.Dimensions[1] + j] = sum;
		}
	}

	return out;
}

//matrix scalar multiplication
Matrix Matrix::operator*(const float& f)
{
	Matrix out(this->Dimensions[0], this->Dimensions[1]);
	for (int i = 0; i < this->Dimensions[0]; i++)
	{
		for (int j = 0; j < this->Dimensions[1]; j++)
		{
			out.Elements[i * Dimensions[1] + j] = this->Elements[i * Dimensions[1] + j] * f;
		}
	}
	return out;
}

//matrix addition
Matrix Matrix::operator+(const Matrix& m)
{
	Matrix out(this->Dimensions[0], this->Dimensions[1]);
	if (this->Dimensions[0] != m.Dimensions[0] || this->Dimensions[1] != m.Dimensions[1])
	{
		std::cout << "Error: Invalid matrix addition operation " <<
			this->Dimensions[0] << "x" << this->Dimensions[1] << " * " <<
			m.Dimensions[0] << "x" << m.Dimensions[1] << std::endl;
		return out;
	}
	for (int i = 0; i < this->Dimensions[0]; i++)
	{
		for (int j = 0; j < this->Dimensions[1]; j++)
		{
			out.Elements[i * out.Dimensions[1] + j] = this->Elements[i * this->Dimensions[1] + j]
				+ m.Elements[i * m.Dimensions[1] + j];
		}
	}
	return out;
}

//matrix transpose
Matrix Matrix::Transpose()
{
	Matrix out(Dimensions[1], Dimensions[0]);
	for (int i = 0; i < Dimensions[0]; i++)
	{
		for (int j = 0; j < Dimensions[1]; j++)
		{
			out.Elements[j * out.Dimensions[1] + i] = Elements[i * Dimensions[1] + j];
		}
	}

	return out;
}

//print matrix to console
void Matrix::CoutMatrix()
{
	for (int i = 0; i < this->Dimensions[0]; i++)
	{
		for (int j = 0; j < this->Dimensions[1]; j++)
		{
			std::cout << this->Elements[i * this->Dimensions[1] + j] << ", ";
		}
		std::cout << "\n";
	}
}
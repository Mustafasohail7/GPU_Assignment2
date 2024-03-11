#ifndef ACTIVATION_INC
#define ACTIVATION_INC
#pragma once
#include <cmath>
#include <Eigen/Dense>

extern Eigen::MatrixXf HostMatrixMultiplication(const Eigen::MatrixXf &M, const Eigen::MatrixXf &N);
extern Eigen::MatrixXf HostMatrixScalarMultiplication(const Eigen::MatrixXf &M, float scalar);
extern Eigen::MatrixXf HostMatrixAddition(const Eigen::MatrixXf &M, const Eigen::MatrixXf &N);
extern Eigen::MatrixXf HostMatrixSubtraction(const Eigen::MatrixXf &M, const Eigen::MatrixXf &N);

//activation functions
float sigmoid(float x)
{
	return 1.0f / 1.0f + exp(-x);
}

float sigmoid_prime(float x)
{
	float s = sigmoid(x);
	return s * (1 - s);
}
float tanh2(float x)
{
	return tanh(x);
}

float tanh_prime(float x)
{
	return 1.0f - powf(tanh(x), 2.0f);
}

float relu(float x)
{
	return std::max(x, 0.0f);
}
float relu_prime(float x)
{
	return (float)((int)(x >= 0));
}

float one_minus(float x)
{
	return 1 - x;
}
//loss function and their derivative
float mse(Eigen::MatrixXf& y_true, Eigen::MatrixXf& y_pred)
{
  // Calculate on GPU
	return HostMatrixMultiplication(HostMatrixSubtraction(y_true, y_pred), HostMatrixSubtraction(y_true, y_pred).transpose()).mean();
}

Eigen::MatrixXf mse_prime(Eigen::MatrixXf& y_true, Eigen::MatrixXf& y_pred)
{
  // Calculate on GPU
  return HostMatrixScalarMultiplication(HostMatrixScalarMultiplication(HostMatrixSubtraction(y_pred, y_true), 2), 1 / (y_true.rows() * y_true.cols()));
	// return  2 * (y_pred - y_true) / (y_true.rows()*y_true.cols());
}

float binary_cross_entropy(Eigen::MatrixXf& y_true, Eigen::MatrixXf& y_pred)
{
  // Calculate on GPU
  return HostMatrixMultiplication(HostMatrixSubtraction(HostMatrixScalarMultiplication(y_true, -1), HostMatrixMultiplication(y_pred, HostMatrixScalarMultiplication(y_true, -1)).log()), HostMatrixSubtraction(HostMatrixScalarMultiplication(HostMatrixScalarMultiplication(y_true, -1), HostMatrixMultiplication(y_pred, HostMatrixScalarMultiplication(y_true, -1)).log()), HostMatrixScalarMultiplication(HostMatrixScalarMultiplication(y_true, -1), HostMatrixMultiplication(y_pred, HostMatrixScalarMultiplication(y_true, -1)).log())).transpose()).mean();
}

#endif
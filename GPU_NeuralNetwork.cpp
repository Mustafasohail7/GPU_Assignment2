#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <vector>
#include <functional>
#include <Eigen/Dense>
#include <algorithm>
#include <cuda_runtime.h>

__global__ void forwardKernel(float* input, float* weights, float* bias, float* output, int inputSize, int outputSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < outputSize) {
        float sum = 0.0;
        for (int j = 0; j < inputSize; ++j) {
            sum += input[j] * weights[j * outputSize + idx];
        }
        output[idx] = sum + bias[idx];
    }
}


class Layer {
public:
    Layer() :input(), output() {}
    virtual ~Layer() {}

    virtual void forwardPropagation(const Eigen::MatrixXf& input, Eigen::MatrixXf& output) = 0;
    virtual void backwardPropagation(const Eigen::MatrixXf& outputError, const float learningRate, Eigen::MatrixXf& inputError) = 0;
};



class DenseLayer : public Layer {
private:
    float *d_weights; // Device pointer for weights
    float *d_bias;    // Device pointer for bias
    int inputSize, outputSize;

public:
    DenseLayer(int inputSize, int outputSize) : inputSize(inputSize), outputSize(outputSize) {
        // Allocate memory on device for weights and biases
        cudaMalloc(&d_weights, inputSize * outputSize * sizeof(float));
        cudaMalloc(&d_bias, outputSize * sizeof(float));

        // Here you would initialize the weights and biases on the host and then copy them to the device
        // For simplicity, we'll assume they're already initialized to some values here
    }

    void forwardPropagation(const float* input, float* output) {
        // Assuming 'input' and 'output' are pointers to device memory allocated by the caller

        // Calculate grid and block sizes
        int blockSize = 256; // Choose based on your GPU's architecture
        int numBlocks = (outputSize + blockSize - 1) / blockSize;

        // Launch the kernel
        forwardKernel<<<numBlocks, blockSize>>>(input, d_weights, d_bias, output, inputSize, outputSize);
        
        // In real code, check for errors after kernel launch
        cudaDeviceSynchronize(); // Wait for the kernel to complete
    }

    ~DenseLayer() {
        // Free device memory
        cudaFree(d_weights);
        cudaFree(d_bias);
    }
};

#ifndef ACTIVATION_CUDA_INC
#define ACTIVATION_CUDA_INC
#pragma once

#include <cmath>

__device__ float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__device__ float sigmoid_prime(float x)
{
    float s = sigmoid(x);
    return s * (1 - s);
}

__device__ float tanh2(float x)
{
    return tanhf(x);
}

__device__ float tanh_prime(float x)
{
    float tanhx = tanhf(x);
    return 1.0f - tanhx * tanhx;
}

__device__ float relu(float x)
{
    return fmaxf(x, 0.0f);
}

__device__ float relu_prime(float x)
{
    return (x >= 0.0f) ? 1.0f : 0.0f;
}

__device__ float one_minus(float x)
{
    return 1.0f - x;
}

#endif

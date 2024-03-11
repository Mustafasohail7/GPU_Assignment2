#include <iostream>
#include <Eigen/Dense>
#include <cuda_runtime.h>

__global__ void MatrixScalarMultiply(float *M, float N, float *P, int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols)
    {
        float result = M[col * rows + row] * N;
        P[col * rows + row] = result;
    }
}

__global__ void DeviceMatrixMultiply(float *M, float *N, float *P, int rows,
                                    int cols, int common)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols)
    {
        float Pvalue = 0;

        for (int k = 0; k < common; ++k)
        {
            float Mvalue = M[k * rows + row];
            float Nvalue = N[col * common + k];
            Pvalue += Mvalue * Nvalue;
        }

        P[col * rows + row] = Pvalue;
    }
}

__global__ void DeviceMatrixAddition(float *M, float *N, float *P, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < cols && row < rows)
    {
        int index = col * rows + row;
        P[index] = M[index] + N[index];
    }
}

__global__ void DeviceMatrixSubtraction(float *M, float *N, float *P, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < cols && row < rows)
    {
        int index = col * rows + row;
        P[index] = M[index] - N[index];
    }
}

Eigen::MatrixXf HostMatrixMultiply(const Eigen::MatrixXf &M, const Eigen::MatrixXf &N)
{
    int rows = M.rows();
    int cols = N.cols();

    int common = M.cols();
    float *d_M, *d_N, *d_P;
    int size_M = rows * common * sizeof(float);
    int size_N = common * cols * sizeof(float);
    int size_P = rows * cols * sizeof(float);

    cudaMalloc((void **)&d_M, size_M);
    cudaMalloc((void **)&d_N, size_N);
    cudaMalloc((void **)&d_P, size_P);
    cudaMemcpy(d_M, M.data(), size_M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N.data(), size_N, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((rows + dimBlock.x - 1) / dimBlock.x, (cols + dimBlock.y - 1) / dimBlock.y);
    DeviceMatrixMultiply<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, rows, cols, common);
    cudaDeviceSynchronize();
    cudaGetLastError();

    Eigen::MatrixXf P(rows, cols);
    cudaMemcpy(P.data(), d_P, size_P, cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return P;
}

Eigen::MatrixXf HostMatrixScalarMultiply(const Eigen::MatrixXf &M, float N)
{
    int rows = M.rows();
    int cols = M.cols();
    float *d_M, *d_P;
    int size_M = rows * cols * sizeof(float);
    int size_P = rows * cols * sizeof(float);
    cudaMalloc((void **)&d_M, size_M);
    cudaMalloc((void **)&d_P, size_P);
    cudaMemcpy(d_M, M.data(), size_M, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((rows + dimBlock.x - 1) / dimBlock.x, (cols + dimBlock.y - 1) / dimBlock.y);
    DeviceMatrixScalarMultiply<<<dimGrid, dimBlock>>>(d_M, N, d_P, rows, cols);
    cudaDeviceSynchronize();
    cudaGetLastError();

    Eigen::MatrixXf P(rows, cols);
    cudaMemcpy(P.data(), d_P, size_P, cudaMemcpyDeviceToHost);
    cudaFree(d_M);
    cudaFree(d_P);

    return P;
}


Eigen::MatrixXf HostMatrixAddition(const Eigen::MatrixXf &M, const Eigen::MatrixXf &N)
{
    int rows = M.rows();
    int cols = M.cols();
    float *d_M, *d_N, *d_P;
    int size_M = rows * cols * sizeof(float);
    int size_P = rows * cols * sizeof(float);
    cudaMalloc((void **)&d_M, size_M);
    cudaMalloc((void **)&d_N, size_M);
    cudaMalloc((void **)&d_P, size_P);
    cudaMemcpy(d_M, M.data(), size_M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N.data(), size_M, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((rows + dimBlock.x - 1) / dimBlock.x, (cols + dimBlock.y - 1) / dimBlock.y);
    DeviceMatrixAddition<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, rows, cols);
    cudaDeviceSynchronize();
    cudaGetLastError();

    Eigen::MatrixXf P(rows, cols);
    cudaMemcpy(P.data(), d_P, size_P, cudaMemcpyDeviceToHost);
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return P;
}



Eigen::MatrixXf HostMatrixSubtraction(const Eigen::MatrixXf &M, const Eigen::MatrixXf &N)
{
    int rows = M.rows();
    int cols = M.cols();
    float *d_M, *d_N, *d_P;
    int size_M = rows * cols * sizeof(float);
    int size_P = rows * cols * sizeof(float);
    cudaMalloc((void **)&d_M, size_M);
    cudaMalloc((void **)&d_N, size_M);
    cudaMalloc((void **)&d_P, size_P);
    cudaMemcpy(d_M, M.data(), size_M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N.data(), size_M, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((rows + dimBlock.x - 1) / dimBlock.x, (cols + dimBlock.y - 1) / dimBlock.y);
    DeviceMatrixSubtraction<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, rows, cols);
    cudaDeviceSynchronize();
    cudaGetLastError();

    Eigen::MatrixXf P(rows, cols);
    cudaMemcpy(P.data(), d_P, size_P, cudaMemcpyDeviceToHost);
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return P;
}
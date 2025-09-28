#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cstdarg>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int hashI32(int& seed) {
    seed += 0xe6546b64;
    seed ^= seed >> 16;
    seed *= 0x85ebca6b;
    seed ^= seed >> 13;
    seed *= 0xc2b2ae35;
    seed ^= seed >> 16;
    return seed;
}

float hashF32(int& seed, float min = 0.0f, float max = 1.0f) {
    return min + (hashI32(seed) & 0x7fffffff) * (max - min) * 0.0000000004656613f;
}

void printHostTensor(
    const char *name,
    int width, int height,
    const float *hostTensor, int stride,
    ...
) {
    va_list args;
    va_start(args, stride);
    vprintf(name, args);
    va_end(args);
    printf(":\n");
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf("%.3f ", hostTensor[y * width + x]);
        }
        printf("\n");
    }
}

void printDeviceTensor(
    const char *name,
    int width, int height,
    const float *deviceTensor, int stride,
    ...
) {
    float *hostTensor = (float*)malloc(width * height * sizeof(float));
    cudaMemcpy2D(
        hostTensor, width * sizeof(float),
        deviceTensor, stride * sizeof(float),
        width * sizeof(float), height,
        cudaMemcpyDeviceToHost
    );
    va_list args;
    va_start(args, stride);
    vprintf(name, args);
    va_end(args);
    printf(":\n");
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf("%.3f ", hostTensor[y * width + x]);
        }
        printf("\n");
    }
    free(hostTensor);
}

__global__ void randNormalKernel(
    int width, int height,
    float *dTensor, int stride,
    int seed, float mean, float std
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;
    seed += idx * 0xe6546b64;
    seed ^= seed >> 16;
    seed *= 0x85ebca6b;
    seed ^= seed >> 13;
    seed *= 0xc2b2ae35;
    seed ^= seed >> 16;
    int y = idx / width;
    int x = idx - y * width;
    float u1 = (seed & 0xffff) * 0.0000152587890625f;
    float u2 = ((seed >> 16) & 0xffff) * 0.0000152587890625f;
    float r = sqrtf(-2.0f * logf(u1 + 1e-8f)) * cosf(6.283185307179586f * u2);
    dTensor[y * stride + x] = r * std + mean;
}

void normalRandFill(
    int width, int height,
    float *dTensor, int stride,
    int &seed, float mean, float std
) {
    int blockSize = 256;
    int gridSize = (width * height + blockSize - 1) / blockSize;
    randNormalKernel<<<gridSize, blockSize>>>(
        width, height,
        dTensor, stride,
        hashI32(seed), mean, std
    );
}

__global__ void randUniformKernel(
    int width, int height,
    float *dTensor, int stride,
    int seed, float min, float max
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;
    seed += idx * 0xe6546b64;
    seed ^= seed >> 16;
    seed *= 0x85ebca6b;
    seed ^= seed >> 13;
    seed *= 0xc2b2ae35;
    seed ^= seed >> 16;
    int y = idx / width;
    int x = idx - y * width;
    float u1 = (seed & 0xffff) * 0.0000152587890625f;
    dTensor[y * stride + x] = u1 * (max - min) + min;
}

void uniformRandFill(
    int width, int height,
    float *dTensor, int stride,
    int &seed, float min = 0.0f, float max = 1.0f
) {
    int blockSize = 256;
    int gridSize = (width * height + blockSize - 1) / blockSize;
    randUniformKernel<<<gridSize, blockSize>>>(
        width, height,
        dTensor, stride,
        hashI32(seed), min, max
    );
}

template<typename T>
T* mallocDeviceTensor(int size) {
    T *deviceTensor;
    cudaMalloc(&deviceTensor, size * sizeof(T));
    return deviceTensor;
}

template<typename T>
T* callocDeviceTensor(int size) {
    T *deviceTensor = mallocDeviceTensor<T>(size);
    cudaMemset(deviceTensor, 0, size * sizeof(T));
    return deviceTensor;
}

__global__ void batchNormKernel(
    int width, int height,
    const float *dInput, int inputStride, int inputBatchStride,
    float *dNorm, int normStride, int normBatchStride
) {
    __shared__ float sTmp[32];
    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    int batch = blockIdx.x / width;
    int x = blockIdx.x - width * batch;
    float val = threadIdx.x < height ? dInput[batch * inputBatchStride + threadIdx.x * inputStride + x] : 0.0f;
    float sum = val * val;
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, i);
    if (lane == 0) sTmp[warpId] = sum;
    __syncthreads();
    if (warpId == 0) {
        sum = sTmp[lane];
        #pragma unroll
        for (int i = 16; i > 0; i >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, i);
        if (lane == 0) sTmp[0] = rsqrtf(sum / height + 1e-8f);
    }
    __syncthreads();
    if (threadIdx.x < height) dNorm[batch * normBatchStride + threadIdx.x * normStride + x] = val * sTmp[0];
}

void batchNorm(
    int width, int height,
    const float *dInput, int inputStride, int inputBatchStride,
    float *dNorm, int normStride, int normBatchStride,
    int batches
) {
    assert(height <= 1024);
    int blockSize = 1024;
    int gridSize = width * batches;
    batchNormKernel<<<gridSize, blockSize>>>(
        width, height,
        dInput, inputStride, inputBatchStride,
        dNorm, normStride, normBatchStride
    );
}

__global__ void batchNormGradKernel(
    int width, int height,
    float *dNormal, int normalStride, int normalBatchStride,
    float *dInput, int inputStride, int inputBatchStride,
    float *dOutputGrad, int outputGradStride, int outputGradBatchStride
) {
    __shared__ float sTmp[32];
    __shared__ float sTmp2[32];
    int tid = threadIdx.x;
    uint8_t lane = tid & 31;
    uint8_t warpId = tid >> 5;
    int batch = blockIdx.x / width;
    int x = blockIdx.x - width * batch;
    float val = (tid < height) ? dInput[batch * inputBatchStride + tid * inputStride + x] : 0.0f;
    float grad = (tid < height) ? dNormal[batch * normalBatchStride + tid * normalStride + x] : 0.0f;
    float sum = val * val;
    float sum2 = grad * val;
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, i);
        sum2 += __shfl_down_sync(0xFFFFFFFF, sum2, i);
    }
    if (lane == 0) {
        sTmp[warpId] = sum;
        sTmp2[warpId] = sum2;
    }
    __syncthreads();
    if (warpId == 0) {
        sum = sTmp[lane];
        sum2 = sTmp2[lane];
        #pragma unroll
        for (int i = 16; i > 0; i >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, i);
            sum2 += __shfl_down_sync(0xFFFFFFFF, sum2, i);
        }
        if (lane == 0) {
            sTmp[0] = rsqrtf(sum / height + 1e-8f);
            sTmp2[0] = sum2 / height;
        }
    }
    __syncthreads();
    if (tid < height) {
        dOutputGrad[batch * outputGradBatchStride + tid * outputGradStride + x] +=
            (grad - val * sTmp2[0] * sTmp[0] * sTmp[0]) * sTmp[0];
        dNormal[batch * normalBatchStride + tid * normalStride + x] = val * sTmp[0];
    }
}

void batchNormGrad(
    int width, int height,
    float *dNormal, int normalStride, int normalBatchStride,
    float *dInput, int inputStride, int inputBatchStride,
    float *dOutputGrad, int outputGradStride, int outputGradBatchStride,
    int batches
) {
    assert(height <= 1024);
    int blockSize = 1024;
    int gridSize = width * batches;
    batchNormGradKernel<<<gridSize, blockSize>>>(
        width, height,
        dNormal, normalStride, normalBatchStride,
        dInput, inputStride, inputBatchStride,
        dOutputGrad, outputGradStride, outputGradBatchStride
    );
}

__global__ void residualSwigluKernel(
    int width, int height,
    float *dInput, int inputStride, int inputBatchStride,
    float *dLogit, int logitStride, int logitBatchStride,
    float *dResidual, int residualStride, int residualBatchStride,
    int batches
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height * batches) return;
    int y = idx / width;
    int x = idx - y * width;
    int batch = y / height;
    y -= batch * height;
    float val = dInput[batch * inputBatchStride + y * inputStride + x];
    float logit = dLogit[batch * logitBatchStride + y * logitStride + x];
    float bias = dResidual[batch * residualBatchStride + y * residualStride + x];
    float sigmoid = 1.0f / (1.0f + expf(-logit));
    dInput[batch * inputBatchStride + y * inputStride + x] = val * sigmoid + bias;
    dLogit[batch * logitBatchStride + y * logitStride + x] = sigmoid;
}

void residualSwiglu(
    int width, int height,
    float *dInput, int inputStride, int inputBatchStride,
    float *dLogit, int logitStride, int logitBatchStride,
    float *dResidual, int residualStride, int residualBatchStride,
    int batches
) {
    int blockSize = 256;
    int gridSize = (width * height * batches + blockSize - 1) / blockSize;
    residualSwigluKernel<<<gridSize, blockSize>>>(
        width, height,
        dInput, inputStride, inputBatchStride,
        dLogit, logitStride, logitBatchStride,
        dResidual, residualStride, residualBatchStride,
        batches
    );
}

__global__ void residualSwigluGradKernel(
    int width, int height,
    float *dOutputGrad, int outputGradStride, int outputGradBatchStride,
    float *dLogitGrad, int logitGradStride, int logitGradBatchStride,
    float *dResidualGrad, int residualGradStride, int residualGradBatchStride,
    float *dOutput, int outputStride, int outputBatchStride,
    float *dSigmoid, int sigmoidStride, int sigmoidBatchStride,
    float *dResidual, int residualStride, int residualBatchStride,
    int batches
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height * batches) return;
    int y = idx / width;
    int x = idx - y * width;
    int batch = y / height;
    y -= batch * height;
    float swiglu = dOutput[batch * outputBatchStride + y * outputStride + x];
    float sigmoid = dSigmoid[batch * sigmoidBatchStride + y * sigmoidStride + x];
    float bias = dResidual[batch * residualBatchStride + y * residualStride + x];
    float grad = dOutputGrad[batch * outputGradBatchStride + y * outputGradStride + x];
    dOutputGrad[batch * outputGradBatchStride + y * outputGradStride + x] = grad * sigmoid;
    dLogitGrad[batch * logitGradBatchStride + y * logitGradStride + x] = grad * (swiglu - bias) * (1.0f - sigmoid);
    dResidualGrad[batch * residualGradBatchStride + y * residualGradStride + x] = grad;
}

void residualSwigluGrad(
    int width, int height,
    float *dOutputGrad, int outputGradStride, int outputGradBatchStride,
    float *dLogitGrad, int logitGradStride, int logitGradBatchStride,
    float *dResidualGrad, int residualGradStride, int residualGradBatchStride,
    float *dOutput, int outputStride, int outputBatchStride,
    float *dSigmoid, int sigmoidStride, int sigmoidBatchStride,
    float *dResidual, int residualStride, int residualBatchStride,
    int batches
) {
    int blockSize = 256;
    int gridSize = (width * height * batches + blockSize - 1) / blockSize;
    residualSwigluGradKernel<<<gridSize, blockSize>>>(
        width, height,
        dOutputGrad, outputGradStride, outputGradBatchStride,
        dLogitGrad, logitGradStride, logitGradBatchStride,
        dResidualGrad, residualGradStride, residualGradBatchStride,
        dOutput, outputStride, outputBatchStride,
        dSigmoid, sigmoidStride, sigmoidBatchStride,
        dResidual, residualStride, residualBatchStride,
        batches
    );
}

__global__ void adamUpdateKernel(
    int width, int height,
    const float *dWeightGradient, int gradStride,
    float *dWeightGradMean, int gradMeanStride,
    float *dWeightGradVar, int gradVarStride,
    float *dWeights, int weightStride,
    float lr, float meanBeta, float varBeta, float epsilon
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;
    int y = idx / width;
    int x = idx - y * width;
    int meanIdx = y * gradMeanStride + x;
    int varIdx = y * gradVarStride + x;
    float grad = dWeightGradient[y * gradStride + x];
    float mean = meanBeta * dWeightGradMean[meanIdx] + (1.0f - meanBeta) * grad;
    float var = varBeta * dWeightGradVar[varIdx] + (1.0f - varBeta) * grad * grad;
    dWeightGradMean[meanIdx] = mean;
    dWeightGradVar[varIdx] = var;
    dWeights[y * weightStride + x] += lr * mean * rsqrtf(var + epsilon);
}

void adamUpdate(
    int width, int height,
    const float *dWeightGradient, int gradStride,
    float *dWeightGradMean, int gradMeanStride,
    float *dWeightGradVar, int gradVarStride,
    float *dWeights, int weightStride,
    float lr, float meanBeta, float varBeta, float epsilon
) {
    int blockSize = 256;
    int gridSize = (width * height + blockSize - 1) / blockSize;
    adamUpdateKernel<<<gridSize, blockSize>>>(
        width, height,
        dWeightGradient, gradStride,
        dWeightGradMean, gradMeanStride,
        dWeightGradVar, gradVarStride,
        dWeights, weightStride,
        lr, meanBeta, varBeta, epsilon
    );
}
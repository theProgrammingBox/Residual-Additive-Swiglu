#pragma once

#include <stdio.h>
#include <cstdarg>
#include <assert.h>
#include <cublas_v2.h>

// =====================================================================================
// === Inline and Template Definitions (These MUST stay in the header file) ==========
// =====================================================================================

inline int hashI32(int& seed) {
    seed += 0xe6546b64;
    seed ^= seed >> 16;
    seed *= 0x85ebca6b;
    seed ^= seed >> 13;
    seed *= 0xc2b2ae35;
    seed ^= seed >> 16;
    return seed;
}

inline float hashF32(int& seed, float min = 0.0f, float max = 1.0f) {
    return min + (hashI32(seed) & 0x7fffffff) * (max - min) * 0.0000000004656613f;
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

// =====================================================================================
// === Function and Kernel Declarations (The "API") ====================================
// =====================================================================================

void saveDeviceTensor(const char* filename, const float* dTensor, size_t size);
void loadDeviceTensor(const char* filename, float* dTensor, size_t size);

void printHostTensor(const char *name, int width, int height, const float *hostTensor, int stride, ...);
void printDeviceTensor(const char *name, int width, int height, const float *deviceTensor, int stride, ...);

void normalRandFill(int width, int height, float *dTensor, int stride, int &seed, float mean, float std);
void uniformRandFill(int width, int height, float *dTensor, int stride, int &seed, float min = 0.0f, float max = 1.0f);
void generalIdentityFill(int width, int height, float *dTensor, int stride, int batchStride, int batches);
void positionEmbedding(int biasBits, int temporalBits, int batches, int sequences, float* dTensor, int stride);
void batchNorm(int width, int height, const float *dInput, int inputStride, int inputBatchStride, float *dNorm, int normStride, int normBatchStride, int batches);
void batchNormGrad(int width, int height, float *dNormal, int normalStride, int normalBatchStride, float *dInput, int inputStride, int inputBatchStride, float *dOutputGrad, int outputGradStride, int outputGradBatchStride, int batches);
void softmax(int width, int height, const float *dInput, int inputStride, float *dOutput, int outputStride, int mask = 0);
void scoreMask(int batches, int sequences, const float *dScores, int scoresSeqStride, int scoresBatchStride, float *dOutput, int outputSeqStride, int outputBatchStride);
void softmaxGrad(int width, int height, const float* dSoftmax, int softmaxStride, const float* dSoftmaxGrad, int gradStride, float* dLogitGrad, int logitStride);
void softmaxSample(int width, int height, const float *dTensor, int stride, float *dOutput, int outputStride, int *dSamples, int &seed);
void residualSwiglu(int width, int height, float *dInput, int inputStride, int inputBatchStride, float *dLogit, int logitStride, int logitBatchStride, float *dResidual, int residualStride, int residualBatchStride, int batches);

// THIS IS THE LINE THAT IS NOW FIXED
void residualSwigluGrad(int width, int height, float *dOutputGrad, int outputGradStride, int outputGradBatchStride, float *dLogitGrad, int logitGradStride, int logitGradBatchStride, float *dResidualGrad, int residualGradStride, int resGradBatchStride, float *dOutput, int outputStride, int outputBatchStride, float *dSigmoid, int sigmoidStride, int sigmoidBatchStride, float *dResidual, int residualStride, int resBatchStride, int batches);

void adamUpdate(int width, int height, const float *dWeightGradient, int gradStride, float *dWeightGradMean, int gradMeanStride, float *dWeightGradVar, int gradVarStride, float *dWeights, int weightStride, float lr, float meanBeta, float varBeta, float epsilon);
void swap2D(int width, int height, float* dTensorA, int strideA, float* dTensorB, int strideB);
void gaeGradient(int batches, int sequences, int actions, const float* dValue, int valueStride, const float* dProb, const float* dReward, const int* dSamples, float* dOutputGrad, int outputGradStride, float DISCOUNT, float LAMBDA, float entropy_beta);
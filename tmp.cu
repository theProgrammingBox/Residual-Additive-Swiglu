#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include <cublas_v2.h>
#include <cudnn.h>

void failIf(uint8_t condition, const char* message) {
    if (condition) {
        printf("%s\n", message);
        exit(1);
    }
}

float* cudaMallocF32(uint32_t size) {
    float* dTensor;
    cudaError_t status = cudaMalloc((void**)&dTensor, size * sizeof(float));
    failIf(status != cudaSuccess, "ERROR: cudaMalloc failed");
    return dTensor;
}

float* mallocF32(uint32_t size) {
    float* hTensor = (float*)malloc(size * sizeof(float));
    failIf(hTensor == NULL, "ERROR: malloc failed");
    return hTensor;
}

void printHostTensorF32(float* hTensor, uint32_t width, uint32_t height, const char* name) {
    printf("%s:\n", name);
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            printf("%.1f ", hTensor[y * width + x]);
        }
        printf("\n");
    }
}

void printDeviceTensorF32(float* dTensor, uint32_t width, uint32_t height, const char* name) {
    float *hTensor = (float*)malloc(width * height * sizeof(float));
    cudaMemcpy(hTensor, dTensor, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    printHostTensorF32(hTensor, width, height, name);
    free(hTensor);
}

uint32_t genRandI32(uint32_t* seed, uint32_t key = 0xa9022513) {
    key *= 0xcc9e2d51;
    key = (key << 15) | (key >> 17);
    key *= 0x1b873593;
    *seed ^= key;
    *seed = (*seed << 13) | (*seed >> 19);
    *seed = *seed * 5 + 0xe6546b64;
    *seed ^= 4;
    *seed ^= *seed >> 16;
    *seed *= 0x85ebca6b;
    *seed ^= *seed >> 13;
    *seed *= 0xc2b2ae35;
    *seed ^= *seed >> 16;
    return *seed;
}

float genRandF32(uint32_t* seed, float min = 0, float max = 1, uint32_t key = 0xe3ab464f) {
    return min + (max - min) * (genRandI32(seed, key) / (float)0xFFFFFFFF);
}

void genNorm2F32(uint32_t* seed, float* sample1, float* sample2, float mean1 = 0, float logStdDev1 = 0, float mean2 = 0, float logStdDev2 = 0, uint32_t key = 0x3c6ef372) {
    uint32_t rnd = genRandI32(seed, key);
    float angle  = (rnd >> 16) * 6.28318530718f / 65536.f;
    float radius = sqrtf(-2.f * logf((rnd & 0xFFFF) / 65536.f + 1e-8f));
    *sample1 = mean1 + expf(logStdDev1) * radius * cosf(angle);
    *sample2 = mean2 + expf(logStdDev2) * radius * sinf(angle);
}

__global__ void randomizeDeviceTensorKernelF32(uint32_t height, uint32_t width, float* dTensor, uint32_t stride, uint32_t seed, float min, float max) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t threadId = idx << 1;
    if (threadId < height * width) {
        uint32_t y = threadId / width;
        uint32_t x = threadId - y * width;
        uint32_t key = idx * 0xcc9e2d51;
        key = (key << 15) | (key >> 17);
        key *= 0x1b873593;
        seed ^= key;
        seed = (seed << 13) | (seed >> 19);
        seed = seed * 5 + 0xe6546b64;
        seed ^= 4;
        seed ^= seed >> 16;
        seed *= 0x85ebca6b;
        seed ^= seed >> 13;
        seed *= 0xc2b2ae35;
        seed ^= seed >> 16;
        float scale = (max - min) / (float)0xffff;
        dTensor[y * stride + x] = min + scale * (seed & 0xffff);
        dTensor[y * stride + x + 1] = min + scale * (seed >> 16);
    }
}

void randomizeDeviceTensorF32(uint32_t height, uint32_t width, float* dTensor, uint32_t stride, uint32_t* seed, float min = 0, float max = 1, uint32_t key = 0xcdf2c45b) {
    failIf(height * width & 1, "ERROR: height * width must be even");
    randomizeDeviceTensorKernelF32<<<(height * width + 1023) >> 10, 512>>>(height, width, dTensor, stride, genRandI32(seed, key), min, max);
}

float invSqrt(float number) {
	uint32_t i = 0x5F1FFFF9 - (*(uint32_t*)&number >> 1);
	float tmp = *(float*)&i;
	return tmp * 0.703952253f * (2.38924456f - number * tmp * tmp);
}

__device__ float _invSqrt(float number) {
    uint32_t i = 0x5F1FFFF9 - (*(uint32_t*)&number >> 1);
    float tmp = *(float*)&i;
    return tmp * 0.703952253f * (2.38924456f - number * tmp * tmp);
}

__global__ void adamUpdateKernelF32(uint32_t size, float learningRate, float* dTensorGrad, float* dTensorGradMean, float* dTensorGradVar, float* dTensor, float invMeanCor, float invVarCor, float epsilon = 1e-8f) {
    uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId < size) {
        float grad = dTensorGrad[threadId];
        float mean = 0.9f * dTensorGradMean[threadId] + 0.1f * grad;
        float var = 0.999f * dTensorGradVar[threadId] + 0.001f * grad * grad;
        dTensorGradMean[threadId] = mean;
        dTensorGradVar[threadId] = var;
        dTensor[threadId] += learningRate * mean * invMeanCor * _invSqrt(var * invVarCor + epsilon);
    }
}

void adamUpdateF32(uint32_t size, float learningRate, float* dTensorGrad, float* dTensorGradMean, float* dTensorGradVar, float* dTensor, float invMeanCor, float invVarCor, float epsilon) {
    adamUpdateKernelF32<<<size + 1023 >> 10, 1024>>>(size, learningRate, dTensorGrad, dTensorGradMean, dTensorGradVar, dTensor, invMeanCor, invVarCor, epsilon);
}

__global__ void swigluKernelF32(uint32_t height, uint32_t width, float* dInputTensor, float* dBiasTensor, uint32_t stride) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < height * width) {
        uint32_t y = idx / width;
        uint32_t x = idx - y * width;
        
        float sigmoid = 1.0f / (1.0f + expf(dInputTensor[y * stride + x + width]));
        dInputTensor[y * stride + x] = dInputTensor[y * stride + x] * (1.0f - sigmoid) + dBiasTensor[y * stride + x];
        dInputTensor[y * stride + x + width] = sigmoid;
        
        // float prob1 = dInputTensor[y * stride + x + width];
        // float sigmoid = prob1 * (prob1 >= 0 && prob1 <= 1) + (prob1 > 1);
        // dInputTensor[y * stride + x] = dInputTensor[y * stride + x] * (sigmoid) + dBiasTensor[y * stride + x];
        // dInputTensor[y * stride + x + width] = sigmoid;
    }
}

void swigluF32(uint32_t height, uint32_t width, float* dInputTensor, float* dBiasTensor, uint32_t stride) {
    swigluKernelF32<<<(height * width + 1023) >> 10, 1024>>>(height, width, dInputTensor, dBiasTensor, stride);
}

__global__ void swigluGradKernelF32(uint32_t height, uint32_t width, float* dInputTensor, float* dBiasTensor, float* dInputGradTensor, float* dBiasGradTensor, uint32_t stride) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < height * width) {
        uint32_t y = idx / width;
        uint32_t x = idx - y * width;
        dBiasGradTensor[y * stride + x] = dInputGradTensor[y * stride + x];
        
        float tmp = dInputTensor[y * stride + x + width] * dInputGradTensor[y * stride + x];
        dInputGradTensor[y * stride + x] -= tmp;
        dInputGradTensor[y * stride + x + width] = (dInputTensor[y * stride + x] - dBiasTensor[y * stride + x]) * tmp;
        
        // float prob1 = dInputTensor[y * stride + x + width];
        // float actGrad = (prob1 > 0 && prob1 < 1);
        // dInputGradTensor[y * stride + x] = dInputGradTensor[y * stride + x] * dInputTensor[y * stride + x + width];
        // dInputGradTensor[y * stride + x + width] = dInputGradTensor[y * stride + x + width] * (dInputTensor[y * stride + x] - dBiasTensor[y * stride + x]) * actGrad;
    }
}

void swigluGradF32(uint32_t height, uint32_t width, float* dInputTensor, float* dBiasTensor, float* dInputGradTensor, float* dBiasGradTensor, uint32_t stride) {
    swigluGradKernelF32<<<(height * width + 1023) >> 10, 1024>>>(height, width, dInputTensor, dBiasTensor, dInputGradTensor, dBiasGradTensor, stride);
}

__global__ void sigmoidKernelF32(uint32_t height, uint32_t width, float* dTensor, float* dSigmoidTensor, uint32_t stride) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= height * width) return;
    uint32_t y = idx / width;
    uint32_t x = idx - y * width;
    dSigmoidTensor[y * stride + x] = 1.0f / (1.0f + expf(-dTensor[y * stride + x]));
}

void sigmoidF32(uint32_t height, uint32_t width, float* dTensor, float* dSigmoidTensor, uint32_t stride) {
    sigmoidKernelF32<<<height * width + 1023 >> 10, 1024>>>(height, width, dTensor, dSigmoidTensor, stride);
}

__global__ void invSigmoidKernelF32(uint32_t height, uint32_t width, float* dSigmoidTensor, float* dTensor, uint32_t stride) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= height * width) return;
    uint32_t y = idx / width;
    uint32_t x = idx - y * width;
    dTensor[y * stride + x] = -logf(1.0f / dSigmoidTensor[y * stride + x] - 1.0f);
}

void invSigmoidF32(uint32_t height, uint32_t width, float* dSigmoidTensor, float* dTensor, uint32_t stride) {
    sigmoidKernelF32<<<height * width + 1023 >> 10, 1024>>>(height, width, dSigmoidTensor, dTensor, stride);
}

__global__ void biasKernelF32(uint32_t height, uint32_t width, float* dTensor, float* dBias, uint32_t stride, float invStep) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;
    uint32_t y = idx / width;
    uint32_t x = idx - y * width;
    uint32_t batch = y * invStep;
    dTensor[y * stride + x] += dBias[batch * width + x];
}

void biasF32(uint32_t height, uint32_t width, float* dTensor, float* dBias, uint32_t stride, float invStep) {
    biasKernelF32<<<height, 1024>>>(height, width, dTensor, dBias, stride, invStep);
}

__global__ void biasReductionKernelF32(uint32_t height, uint32_t width, float* dTensor, float* dBiasGrad, uint32_t stride, uint32_t widthStride) {
    __shared__ float sharedTmp[32];
    uint8_t lane = threadIdx.x & 31;
    uint8_t warpId = threadIdx.x >> 5;
    uint32_t y = blockIdx.x / width;
    uint32_t x = blockIdx.x - y * width;
    
    float val = (threadIdx.x < height) ? dTensor[y * stride + threadIdx.x * widthStride + x] : 0;
    float tmp = val;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) tmp += __shfl_down_sync(0xffffffff, tmp, offset);
    if (lane == 0) sharedTmp[warpId] = tmp;
    __syncthreads();
    if (warpId == 0) {
        tmp = sharedTmp[lane];
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) tmp += __shfl_down_sync(0xffffffff, tmp, offset);
        if (lane == 0) sharedTmp[0] = tmp;
    }
    __syncthreads();
    if (lane == 0 && warpId == 0) dBiasGrad[y * width + x] = tmp;
}

void biasReductionF32(uint32_t height, uint32_t width, float* dTensor, float* dBiasGrad, uint32_t sequences, uint32_t stride, uint32_t widthStride) {
    failIf(sequences > 1024, "ERROR: sequences must be <= 1024");
    biasReductionKernelF32<<<width * sequences, 1024>>>(height, width, dTensor, dBiasGrad, stride, widthStride);
}

__global__ void sampleSigmoidKernelF32(uint32_t height, uint32_t width, float* dSigmoidTensor, uint32_t sigmoidStride, float* dSamples, uint32_t sampleStride, uint32_t seed) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t threadId = idx << 1;
    if (threadId < height * width) {
        uint32_t y = threadId / width;
        uint32_t x = threadId - y * width;
        uint32_t key = idx * 0xcc9e2d51;
        key = (key << 15) | (key >> 17);
        key *= 0x1b873593;
        seed ^= key;
        seed = (seed << 13) | (seed >> 19);
        seed = seed * 5 + 0xe6546b64;
        seed ^= 4;
        seed ^= seed >> 16;
        seed *= 0x85ebca6b;
        seed ^= seed >> 13;
        seed *= 0xc2b2ae35;
        seed ^= seed >> 16;
        float prob1 = dSigmoidTensor[y * sigmoidStride + x];
        float prob2 = dSigmoidTensor[y * sigmoidStride + x + 1];
        prob1 = prob1 * (prob1 >= 0 && prob1 <= 1) + (prob1 > 1);
        prob2 = prob2 * (prob2 >= 0 && prob2 <= 1) + (prob2 > 1);
        // prob1 = 1.0f / (1.0f + expf(-prob1));
        // prob2 = 1.0f / (1.0f + expf(-prob2));
        dSamples[y * sampleStride + x] = (prob1 >= (seed & 0xffff) / (float)0xffff) ? 1.0f : 0.0f;
        dSamples[y * sampleStride + x + 1] = (prob2 >= (seed >> 16) / (float)0xffff) ? 1.0f : 0.0f;
    }
}

void sampleSigmoidF32(uint32_t height, uint32_t width, float* dSigmoidTensor, uint32_t sigmoidStride, float* dSamples, uint32_t sampleStride, uint32_t* seed, uint32_t key = 0x3c6ef372) {
    failIf(height * width & 1, "ERROR: height * width must be even");
    sampleSigmoidKernelF32<<<(height * width + 1023) >> 10, 512>>>(height, width, dSigmoidTensor, sigmoidStride, dSamples, sampleStride, genRandI32(seed, key));
}

__global__ void computeSigmoidSampleGradsKernelF32(uint32_t height, uint32_t width, float* dLogitGrads, uint32_t dLogitGradStride, float* dRewards, uint32_t rewardStride, float* dSamples, uint32_t sampleStride, float* dLogits, uint32_t logitStride) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < height * width) {
        uint32_t y = idx / width;
        uint32_t x = idx - y * width;
        float reward = dRewards[y * rewardStride + x];
        float sample = dSamples[y * sampleStride + x];
        float logit = dLogits[y * logitStride + x];
        // logit = 1.0f / (1.0f + expf(-logit));
        dLogitGrads[y * dLogitGradStride + x] = reward * (sample - logit);
    }
}
        
void computeSigmoidSampleGradsF32(uint32_t height, uint32_t width, float* dLogitGrads, uint32_t dLogitGradStride, float* dRewards, uint32_t rewardStride, float* dSamples, uint32_t sampleStride, float* dLogits, uint32_t logitStride) {
    computeSigmoidSampleGradsKernelF32<<<(height * width + 1023) >> 10, 1024>>>(height, width, dLogitGrads, dLogitGradStride, dRewards, rewardStride, dSamples, sampleStride, dLogits, logitStride);
}

__global__ void reluKernelF32(uint32_t height, uint32_t width, float* dTensor, float* dReluTensor, uint32_t stride) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < height * width) {
        uint32_t y = idx / width;
        uint32_t x = idx - y * width;
        dReluTensor[y * stride + x] *= (dTensor[y * stride + x] > 0);
    }
}

void reluF32(uint32_t height, uint32_t width, float* dTensor, float* dReluTensor, uint32_t stride) {
    reluKernelF32<<<height * width + 1023 >> 10, 1024>>>(height, width, dTensor, dReluTensor, stride);
}

__global__ void reluGradKernelF32(uint32_t height, uint32_t width, float* dReluTensor, float* dTensorGrad, uint32_t stride) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < height * width) {
        uint32_t y = idx / width;
        uint32_t x = idx - y * width;
        dTensorGrad[y * stride + x] *= (dReluTensor[y * stride + x] > 0);
    }
}

void reluGradF32(uint32_t height, uint32_t width, float* dReluTensor, float* dTensorGrad, uint32_t stride) {
    reluGradKernelF32<<<height * width + 1023 >> 10, 1024>>>(height, width, dReluTensor, dTensorGrad, stride);
}

#define EPOCHES (1024 * 2)
#define LOG_SKIPS (64 * 2)

#define INPUTS (32)
#define HIDDENS (256)
#define COMPRESS_DIM (256)
#define OUTPUTS (32)
#define LAYERS (32)
#define BATCHES (1024)
#define SEQUENCES (1)

#define WEIGHT_LEARNING_RATE (0.1f * invSqrt(SEQUENCES * BATCHES * HIDDENS))
#define BIAS_LEARNING_RATE (WEIGHT_LEARNING_RATE)
#define MEAN_BETA (0.9f)
#define VAR_BETA (0.999f)
#define EPSILON (1e-8f)
#define BIAS_INV_STEP (1.0f / BATCHES)

int main() {
    failIf(INPUTS != 32 || OUTPUTS != 32, "ERROR: INPUTS and OUTPUTS must be 32");
    failIf(HIDDENS < 32, "ERROR: HIDDENS must be at least 32");
    failIf(LAYERS < 1, "ERROR: LAYERS must be at least 1");
    
    uint32_t seed = time(NULL);
    float N_ONE = -1.0f;
    float ONE = 1.0f;
    float ZERO = 0.0f;
    float meanCor = 0;
    float varCor = 0;
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    
    // init host data
    float *hInputs = mallocF32(BATCHES * INPUTS);
    float *hTargets = mallocF32(SEQUENCES * BATCHES * OUTPUTS);
    float *hSamples = mallocF32(SEQUENCES * BATCHES * OUTPUTS);
    float *hRewards = mallocF32(SEQUENCES * BATCHES * OUTPUTS);
    
    // allocate network
    float *dBiases = cudaMallocF32(SEQUENCES * HIDDENS);
    float *dBiasGrads = cudaMallocF32(SEQUENCES * HIDDENS);
    float *dBiasGradMeans = cudaMallocF32(SEQUENCES * HIDDENS);
    float *dBiasGradVars = cudaMallocF32(SEQUENCES * HIDDENS);
    
    float *dWeights = cudaMallocF32(LAYERS * HIDDENS * (2 * HIDDENS));
    float *dWeightGrads = cudaMallocF32(LAYERS * HIDDENS * (2 * HIDDENS));
    float *dWeightGradMeans = cudaMallocF32(LAYERS * HIDDENS * (2 * HIDDENS));
    float *dWeightGradVars = cudaMallocF32(LAYERS * HIDDENS * (2 * HIDDENS));
    
    float *dLogits = cudaMallocF32((LAYERS + 1) * SEQUENCES * BATCHES * (2 * HIDDENS));
    float *dLogitGrads = cudaMallocF32((LAYERS + 1) * SEQUENCES * BATCHES * (2 * HIDDENS));
    
    // float *dWeight1s = cudaMallocF32(LAYERS * HIDDENS * COMPRESS_DIM);
    // float *dWeight1Grads = cudaMallocF32(LAYERS * HIDDENS * COMPRESS_DIM);
    // float *dWeight1GradMeans = cudaMallocF32(LAYERS * HIDDENS * COMPRESS_DIM);
    // float *dWeight1GradVars = cudaMallocF32(LAYERS * HIDDENS * COMPRESS_DIM);
    
    // float *dWeight2s = cudaMallocF32(LAYERS * COMPRESS_DIM * HIDDENS);
    // float *dWeight2Grads = cudaMallocF32(LAYERS * COMPRESS_DIM * HIDDENS);
    // float *dWeight2GradMeans = cudaMallocF32(LAYERS * COMPRESS_DIM * HIDDENS);
    // float *dWeight2GradVars = cudaMallocF32(LAYERS * COMPRESS_DIM * HIDDENS);
    
    // float *dRelus = cudaMallocF32(LAYERS * SEQUENCES * BATCHES * COMPRESS_DIM);
    // float *dReluGrads = cudaMallocF32(LAYERS * SEQUENCES * BATCHES * COMPRESS_DIM);
    
    // float *dLogits = cudaMallocF32((LAYERS + 1) * SEQUENCES * BATCHES * HIDDENS);
    // float *dLogitGrads = cudaMallocF32((LAYERS + 1) * SEQUENCES * BATCHES * HIDDENS);
    
    
    float *dSamples = cudaMallocF32(SEQUENCES * BATCHES * OUTPUTS);
    float *dRewards = cudaMallocF32(SEQUENCES * BATCHES * OUTPUTS);
    
    // init network params
    cudaMemset(dBiases, 0, SEQUENCES * HIDDENS * sizeof(float));
    cudaMemset(dBiasGradMeans, 0, SEQUENCES * HIDDENS * sizeof(float));
    cudaMemset(dBiasGradVars, 0, SEQUENCES * HIDDENS * sizeof(float));
    randomizeDeviceTensorF32(LAYERS * HIDDENS, 2 * HIDDENS, dWeights, 2 * HIDDENS, &seed, -invSqrt(HIDDENS), invSqrt(HIDDENS));
    cudaMemset(dWeightGradMeans, 0, LAYERS * HIDDENS * (2 * HIDDENS) * sizeof(float));
    cudaMemset(dWeightGradVars, 0, LAYERS * HIDDENS * (2 * HIDDENS) * sizeof(float));
    // randomizeDeviceTensorF32(LAYERS * HIDDENS, COMPRESS_DIM, dWeight1s, COMPRESS_DIM, &seed, -invSqrt(COMPRESS_DIM), invSqrt(COMPRESS_DIM));
    // cudaMemset(dWeight1GradMeans, 0, LAYERS * HIDDENS * COMPRESS_DIM * sizeof(float));
    // cudaMemset(dWeight1GradVars, 0, LAYERS * HIDDENS * COMPRESS_DIM * sizeof(float));
    // randomizeDeviceTensorF32(LAYERS * COMPRESS_DIM, HIDDENS, dWeight2s, HIDDENS, &seed, -invSqrt(HIDDENS), invSqrt(HIDDENS));
    // cudaMemset(dWeight2GradMeans, 0, LAYERS * COMPRESS_DIM * HIDDENS * sizeof(float));
    // cudaMemset(dWeight2GradVars, 0, LAYERS * COMPRESS_DIM * HIDDENS * sizeof(float));
    
    // start training
    time_t start_time, end_time;
    time(&start_time);
    for (uint32_t epoch = 0; epoch < EPOCHES; epoch++) {
        // simulate sequences, but data is not sequential
        for (uint32_t sequence = 0; sequence < SEQUENCES; sequence++) {
            switch (1) {
                case 0: // identity test
                    for (uint32_t batch = 0; batch < BATCHES; batch++) {
                        uint32_t a = genRandI32(&seed);
                        for (uint32_t bit = 0; bit < 32; bit++) {
                            hInputs[batch * INPUTS + bit] = (a >> bit) & 1;
                        }
                    }
                    memcpy(hTargets + sequence * BATCHES * OUTPUTS, hInputs, BATCHES * OUTPUTS * sizeof(float));
                    break;
                case 1: // add test
                    for (uint32_t batch = 0; batch < BATCHES; batch++) {
                        uint16_t a = genRandI32(&seed);
                        uint16_t b = genRandI32(&seed);
                        uint32_t c = a + b;
                        for (uint32_t bit = 0; bit < 16; bit++) {
                            hInputs[batch * INPUTS + bit] = (a >> bit) & 1;
                            hInputs[batch * INPUTS + bit + 16] = (b >> bit) & 1;
                            hTargets[sequence * BATCHES * OUTPUTS + batch * OUTPUTS + bit] = (c >> bit) & 1;
                            hTargets[sequence * BATCHES * OUTPUTS + batch * OUTPUTS + bit + 16] = (c >> (bit + 16)) & 1;
                        }
                    }
                    break;
                case 2: // multiply test
                    for (uint32_t batch = 0; batch < BATCHES; batch++) {
                        uint16_t a = genRandI32(&seed);
                        uint16_t b = genRandI32(&seed);
                        uint32_t c = a * b;
                        for (uint32_t bit = 0; bit < 16; bit++) {
                            hInputs[batch * INPUTS + bit] = (a >> bit) & 1;
                            hInputs[batch * INPUTS + bit + 16] = (b >> bit) & 1;
                            hTargets[sequence * BATCHES * OUTPUTS + batch * OUTPUTS + bit] = (c >> bit) & 1;
                            hTargets[sequence * BATCHES * OUTPUTS + batch * OUTPUTS + bit + 16] = (c >> (bit + 16)) & 1;
                        }
                    }
                    break;
            }
            
            // sequential feed forward
            cudaMemcpy2D(dLogits + sequence * BATCHES * (2 * HIDDENS), 2 * HIDDENS * sizeof(float), hInputs, INPUTS * sizeof(float), INPUTS * sizeof(float), BATCHES, cudaMemcpyHostToDevice);
            // cudaMemcpy2D(
            //     dLogits + sequence * BATCHES * HIDDENS, HIDDENS * sizeof(float),
            //     hInputs, INPUTS * sizeof(float),
            //     INPUTS * sizeof(float), BATCHES, cudaMemcpyHostToDevice);
            for (uint32_t layer = 0; layer < LAYERS; layer++) {
                if (layer == 1) biasF32(BATCHES, HIDDENS, dLogits + (layer * SEQUENCES + sequence) * BATCHES * (2 * HIDDENS), dBiases + sequence * HIDDENS, 2 * HIDDENS, BIAS_INV_STEP);
                cublasGemmEx(
                    cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                    2 * HIDDENS, BATCHES, HIDDENS,
                    &ONE,
                    dWeights + layer * HIDDENS * (2 * HIDDENS), CUDA_R_32F, 2 * HIDDENS,
                    dLogits + (layer * SEQUENCES + sequence) * BATCHES * (2 * HIDDENS), CUDA_R_32F, 2 * HIDDENS,
                    &ZERO,
                    dLogits + ((layer + 1) * SEQUENCES + sequence) * BATCHES * (2 * HIDDENS), CUDA_R_32F, 2 * HIDDENS,
                    CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);
                swigluF32(BATCHES, HIDDENS, dLogits + ((layer + 1) * SEQUENCES + sequence) * BATCHES * (2 * HIDDENS), dLogits + (layer * SEQUENCES + sequence) * BATCHES * (2 * HIDDENS), 2 * HIDDENS);
                
                // if (layer == 1) biasF32(BATCHES, HIDDENS, 
                //     dLogits + (layer * SEQUENCES + sequence) * BATCHES * HIDDENS, 
                //     dBiases + sequence * HIDDENS, HIDDENS, BIAS_INV_STEP);
                
                // cublasGemmEx(
                //     cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                //     COMPRESS_DIM, BATCHES, HIDDENS,
                //     &ONE,
                //     dWeight1s + layer * HIDDENS * COMPRESS_DIM, CUDA_R_32F, COMPRESS_DIM,
                //     dLogits + (layer * SEQUENCES + sequence) * BATCHES * HIDDENS, CUDA_R_32F, HIDDENS,
                //     &ZERO,
                //     dRelus + (layer * SEQUENCES + sequence) * BATCHES * COMPRESS_DIM, CUDA_R_32F, COMPRESS_DIM,
                //     CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);
                
                // reluF32(BATCHES, COMPRESS_DIM,
                //     dRelus + (layer * SEQUENCES + sequence) * BATCHES * COMPRESS_DIM,
                //     dRelus + (layer * SEQUENCES + sequence) * BATCHES * COMPRESS_DIM, COMPRESS_DIM);
                
                // cudaMemcpy2D(
                //     dLogits + ((layer + 1) * SEQUENCES + sequence) * BATCHES * HIDDENS,
                //     HIDDENS * sizeof(float),
                //     dLogits + (layer * SEQUENCES + sequence) * BATCHES * HIDDENS,
                //     HIDDENS * sizeof(float),
                //     HIDDENS * sizeof(float), BATCHES, cudaMemcpyDeviceToDevice);
                
                // cublasGemmEx(
                //     cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                //     HIDDENS, BATCHES, COMPRESS_DIM,
                //     &ONE,
                //     dWeight2s + layer * COMPRESS_DIM * HIDDENS, CUDA_R_32F, HIDDENS,
                //     dRelus + (layer * SEQUENCES + sequence) * BATCHES * COMPRESS_DIM, CUDA_R_32F, COMPRESS_DIM,
                //     &ONE,
                //     dLogits + ((layer + 1) * SEQUENCES + sequence) * BATCHES * HIDDENS, CUDA_R_32F, HIDDENS,
                //     CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);
            }
            // sampleSigmoidF32(BATCHES, OUTPUTS, dLogits + (LAYERS * SEQUENCES + sequence) * BATCHES * (2 * HIDDENS), 2 * HIDDENS, dSamples + sequence * BATCHES * OUTPUTS, OUTPUTS, &seed);
        }
        
        
        
        
        
        
        
        
        // compute traditional gradients
        cudaMemcpy2D(
            dLogitGrads + LAYERS * SEQUENCES * BATCHES * (2 * HIDDENS), 2 * HIDDENS * sizeof(float),
            hTargets, OUTPUTS * sizeof(float), OUTPUTS * sizeof(float),
            SEQUENCES * BATCHES, cudaMemcpyHostToDevice);
        cublasAxpyEx(
            cublasHandle, SEQUENCES * BATCHES * (2 * HIDDENS),
            &N_ONE, CUDA_R_32F,
            dLogits + LAYERS * SEQUENCES * BATCHES * (2 * HIDDENS), CUDA_R_32F, 1,
            dLogitGrads + LAYERS * SEQUENCES * BATCHES * (2 * HIDDENS), CUDA_R_32F, 1,
            CUDA_R_32F);
        cudaMemset2D(dLogitGrads + LAYERS * SEQUENCES * BATCHES * (2 * HIDDENS) + OUTPUTS, 2 * HIDDENS * sizeof(float), 0, (2 * HIDDENS - OUTPUTS) * sizeof(float), SEQUENCES * BATCHES);
        if ((epoch + 1) % LOG_SKIPS == 0) {
            float err;
            cublasNrm2Ex(
                cublasHandle, SEQUENCES * BATCHES * (2 * HIDDENS),
                dLogitGrads + LAYERS * SEQUENCES * BATCHES * (2 * HIDDENS), CUDA_R_32F, 1,
                &err, CUDA_R_32F, CUDA_R_32F);
            printf("Epoch %d: %f\n", epoch + 1, err * invSqrt(SEQUENCES * BATCHES * OUTPUTS));
        }
        
        // cudaMemcpy2D(
        //     dLogitGrads + LAYERS * SEQUENCES * BATCHES * HIDDENS, HIDDENS * sizeof(float),
        //     hTargets, OUTPUTS * sizeof(float), OUTPUTS * sizeof(float),
        //     SEQUENCES * BATCHES, cudaMemcpyHostToDevice);
        // cublasAxpyEx(
        //     cublasHandle, SEQUENCES * BATCHES * HIDDENS,
        //     &N_ONE, CUDA_R_32F,
        //     dLogits + LAYERS * SEQUENCES * BATCHES * HIDDENS, CUDA_R_32F, 1,
        //     dLogitGrads + LAYERS * SEQUENCES * BATCHES * HIDDENS, CUDA_R_32F, 1,
        //     CUDA_R_32F);
        // cudaMemset2D(dLogitGrads + LAYERS * SEQUENCES * BATCHES * HIDDENS + OUTPUTS, 
        //     HIDDENS * sizeof(float), 0, 
        //     (HIDDENS - OUTPUTS) * sizeof(float), SEQUENCES * BATCHES);
        // if ((epoch + 1) % LOG_SKIPS == 0) {
        //     float err;
        //     cublasNrm2Ex(
        //         cublasHandle, SEQUENCES * BATCHES * HIDDENS,
        //         dLogitGrads + LAYERS * SEQUENCES * BATCHES * HIDDENS, CUDA_R_32F, 1,
        //         &err, CUDA_R_32F, CUDA_R_32F);
        //     printf("Epoch %d: %f\n", epoch + 1, err * invSqrt(SEQUENCES * BATCHES * OUTPUTS));
        // }
        
        // simulate rl samples
        if (false) {
            // print rewards
            float avgReward = 0;
            cudaMemcpy(hSamples, dSamples, SEQUENCES * BATCHES * OUTPUTS * sizeof(float), cudaMemcpyDeviceToHost);
            
            for (uint32_t sequence = 0; sequence < SEQUENCES; sequence++) {
                for (uint32_t batch = 0; batch < BATCHES; batch++) {
                    for (uint32_t output = 0; output < OUTPUTS; output++) {
                        float sample = hSamples[sequence * BATCHES * OUTPUTS + batch * OUTPUTS + output];
                        float reward = sample == hTargets[sequence * BATCHES * OUTPUTS + batch * OUTPUTS + output];
                        hRewards[sequence * BATCHES * OUTPUTS + batch * OUTPUTS + output] = reward;
                        avgReward += reward;
                    }
                }
            }
            cudaMemcpy(dRewards, hRewards, SEQUENCES * BATCHES * OUTPUTS * sizeof(float), cudaMemcpyHostToDevice);
            computeSigmoidSampleGradsF32(SEQUENCES * BATCHES, OUTPUTS, dLogitGrads + LAYERS * SEQUENCES * BATCHES * (2 * HIDDENS), 2 * HIDDENS, dRewards, OUTPUTS, dSamples, OUTPUTS, dLogits + LAYERS * SEQUENCES * BATCHES * (2 * HIDDENS), 2 * HIDDENS);
            
            if ((epoch + 1) % LOG_SKIPS == 0) {
                printf("%f\n", avgReward / (SEQUENCES * BATCHES * OUTPUTS));
            }
        }
        
        
        
        
        
        
        
        // parallel back propagation
        // invSigmoidF32(SEQUENCES * BATCHES, OUTPUTS, dLogits + LAYERS * SEQUENCES * BATCHES * (2 * HIDDENS), dLogits + LAYERS * SEQUENCES * BATCHES * (2 * HIDDENS), 2 * HIDDENS);
        for (uint32_t layer = LAYERS; layer--;) {
            swigluGradF32(SEQUENCES * BATCHES, HIDDENS, dLogits + (layer + 1) * SEQUENCES * BATCHES * (2 * HIDDENS), dLogits + layer * SEQUENCES * BATCHES * (2 * HIDDENS), dLogitGrads + (layer + 1) * SEQUENCES * BATCHES * (2 * HIDDENS), dLogitGrads + layer * SEQUENCES * BATCHES * (2 * HIDDENS), 2 * HIDDENS);
            cublasGemmEx(
                cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                HIDDENS, BATCHES * SEQUENCES, 2 * HIDDENS,
                &ONE,
                dWeights + layer * HIDDENS * (2 * HIDDENS), CUDA_R_32F, 2 * HIDDENS,
                dLogitGrads + (layer + 1) * SEQUENCES * BATCHES * (2 * HIDDENS), CUDA_R_32F, 2 * HIDDENS,
                &ONE,
                dLogitGrads + layer * SEQUENCES * BATCHES * (2 * HIDDENS), CUDA_R_32F, 2 * HIDDENS,
                CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);
            cublasGemmEx(
                cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                2 * HIDDENS, HIDDENS, BATCHES * SEQUENCES,
                &ONE,
                dLogitGrads + (layer + 1) * SEQUENCES * BATCHES * (2 * HIDDENS), CUDA_R_32F, 2 * HIDDENS,
                dLogits + layer * SEQUENCES * BATCHES * (2 * HIDDENS), CUDA_R_32F, 2 * HIDDENS,
                &ZERO,
                dWeightGrads + layer * HIDDENS * (2 * HIDDENS), CUDA_R_32F, 2 * HIDDENS,
                CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);
            if (layer == 1) biasReductionF32(BATCHES, HIDDENS, dLogitGrads + layer * SEQUENCES * BATCHES * (2 * HIDDENS), dBiasGrads, SEQUENCES, 2 * HIDDENS * BATCHES, 2 * HIDDENS);
            
            // cublasGemmEx(
            //     cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
            //     HIDDENS, COMPRESS_DIM, BATCHES * SEQUENCES,
            //     &ONE,
            //     dLogitGrads + (layer + 1) * SEQUENCES * BATCHES * HIDDENS, CUDA_R_32F, HIDDENS,
            //     dRelus + layer * SEQUENCES * BATCHES * COMPRESS_DIM, CUDA_R_32F, COMPRESS_DIM,
            //     &ZERO,
            //     dWeight2Grads + layer * COMPRESS_DIM * HIDDENS, CUDA_R_32F, HIDDENS,
            //     CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);
                
            // cublasGemmEx(
            //     cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
            //     COMPRESS_DIM, BATCHES * SEQUENCES, HIDDENS,
            //     &ONE,
            //     dWeight2s + layer * COMPRESS_DIM * HIDDENS, CUDA_R_32F, HIDDENS,
            //     dLogitGrads + (layer + 1) * SEQUENCES * BATCHES * HIDDENS, CUDA_R_32F, HIDDENS,
            //     &ZERO,
            //     dReluGrads + layer * SEQUENCES * BATCHES * COMPRESS_DIM, CUDA_R_32F, COMPRESS_DIM,
            //     CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);
                
            // reluGradF32(BATCHES * SEQUENCES, COMPRESS_DIM,
            //     dRelus + layer * SEQUENCES * BATCHES * COMPRESS_DIM,
            //     dReluGrads + layer * SEQUENCES * BATCHES * COMPRESS_DIM,
            //     COMPRESS_DIM);
                
            // cublasGemmEx(
            //     cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
            //     COMPRESS_DIM, HIDDENS, BATCHES * SEQUENCES,
            //     &ONE,
            //     dReluGrads + layer * SEQUENCES * BATCHES * COMPRESS_DIM, CUDA_R_32F, COMPRESS_DIM,
            //     dLogits + layer * SEQUENCES * BATCHES * HIDDENS, CUDA_R_32F, HIDDENS,
            //     &ZERO,
            //     dWeight1Grads + layer * HIDDENS * COMPRESS_DIM, CUDA_R_32F, COMPRESS_DIM,
            //     CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);
            
            // cudaMemcpy2D(
            //     dLogitGrads + layer * SEQUENCES * BATCHES * HIDDENS, HIDDENS * sizeof(float),
            //     dLogitGrads + (layer + 1) * SEQUENCES * BATCHES * HIDDENS, HIDDENS * sizeof(float),
            //     HIDDENS * sizeof(float), BATCHES * SEQUENCES, cudaMemcpyDeviceToDevice);
                
            // cublasGemmEx(
            //     cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
            //     HIDDENS, BATCHES * SEQUENCES, COMPRESS_DIM,
            //     &ONE,
            //     dWeight1s + layer * HIDDENS * COMPRESS_DIM, CUDA_R_32F, COMPRESS_DIM,
            //     dReluGrads + layer * SEQUENCES * BATCHES * COMPRESS_DIM, CUDA_R_32F, COMPRESS_DIM,
            //     &ONE,
            //     dLogitGrads + layer * SEQUENCES * BATCHES * HIDDENS, CUDA_R_32F, HIDDENS,
            //     CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);
                
            // if (layer == 1) biasReductionF32(BATCHES, HIDDENS, 
            //     dLogitGrads + layer * SEQUENCES * BATCHES * HIDDENS, 
            //     dBiasGrads, SEQUENCES, HIDDENS * BATCHES, HIDDENS);
        }
        meanCor *= MEAN_BETA;
        varCor *= VAR_BETA;
        adamUpdateF32(LAYERS * HIDDENS * (2 * HIDDENS), WEIGHT_LEARNING_RATE, dWeightGrads, dWeightGradMeans, dWeightGradVars, dWeights, 1.0f / (1.0f - meanCor), 1.0f / (1.0f - varCor), EPSILON);
        // adamUpdateF32(SEQUENCES * HIDDENS, BIAS_LEARNING_RATE, dBiasGrads, dBiasGradMeans, dBiasGradVars, dBiases, 1.0f / (1.0f - meanCor), 1.0f / (1.0f - varCor), EPSILON);
        
        // adamUpdateF32(LAYERS * HIDDENS * COMPRESS_DIM, WEIGHT_LEARNING_RATE, 
        //     dWeight1Grads, dWeight1GradMeans, dWeight1GradVars, dWeight1s, 
        //     1.0f / (1.0f - meanCor), 1.0f / (1.0f - varCor), EPSILON);

        // // Update Weight2 parameters
        // adamUpdateF32(LAYERS * COMPRESS_DIM * HIDDENS, WEIGHT_LEARNING_RATE, 
        //             dWeight2Grads, dWeight2GradMeans, dWeight2GradVars, dWeight2s, 
        //             1.0f / (1.0f - meanCor), 1.0f / (1.0f - varCor), EPSILON);

        // // Update Bias parameters
        // adamUpdateF32(SEQUENCES * HIDDENS, BIAS_LEARNING_RATE, 
        //             dBiasGrads, dBiasGradMeans, dBiasGradVars, dBiases, 
        //             1.0f / (1.0f - meanCor), 1.0f / (1.0f - varCor), EPSILON);
    }
    time(&end_time);
    printf("Time: %ld\n", end_time - start_time);
    
    // printDeviceTensorF32(dBiases, HIDDENS, SEQUENCES, "dBiases");
    // printDeviceTensorF32(dWeights, HIDDENS, 2 * HIDDENS, "dWeights");
    
    return 0;
}
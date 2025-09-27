#include "header.cuh"

// Note: This file contains the implementations (definitions) for the functions
// declared in header.cuh. This ensures each function is only defined once.

void saveDeviceTensor(const char* filename, const float* dTensor, size_t size) {
    float* hTensor = (float*)malloc(size * sizeof(float));
    cudaMemcpy(hTensor, dTensor, size * sizeof(float), cudaMemcpyDeviceToHost);
    FILE* file = fopen(filename, "wb");
    if (file) {
        fwrite(hTensor, sizeof(float), size, file);
        fclose(file);
    }
    free(hTensor);
}

void loadDeviceTensor(const char* filename, float* dTensor, size_t size) {
    float* hTensor = (float*)malloc(size * sizeof(float));
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        free(hTensor);
        return;
    }
    // Fix for the compiler warning about unused result.
    (void)fread(hTensor, sizeof(float), size, file);
    fclose(file);
    cudaMemcpy(dTensor, hTensor, size * sizeof(float), cudaMemcpyHostToDevice);
    free(hTensor);
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
    int &seed, float min, float max
) {
    int blockSize = 256;
    int gridSize = (width * height + blockSize - 1) / blockSize;
    randUniformKernel<<<gridSize, blockSize>>>(
        width, height,
        dTensor, stride,
        hashI32(seed), min, max
    );
}

__global__ void generalIdentityColumnsKernel(
    int width, int height,
    float *dTensor, int stride, int batchStride,
    int batches
) {
    int colIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (colIndex >= width * batches) return;
    int batch = colIndex / width;
    colIndex -= width * batch;
    dTensor += batch * batchStride;

    float rowsPerColumn = height / (float)width;
    int baseRow = colIndex * rowsPerColumn;
    float shareNextRow = colIndex - (baseRow + 1) / rowsPerColumn + 1.0f;
    shareNextRow *= shareNextRow > 0.0f;
    dTensor[(baseRow + (baseRow + 1 != height)) * stride + colIndex] = shareNextRow;
    dTensor[baseRow * stride + colIndex] = 1.0f - shareNextRow;
}

__global__ void generalIdentityRowsKernel(
    int width, int height,
    float *dTensor, int stride, int batchStride,
    int batches
) {
    int rowIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (rowIndex >= height * batches) return;
    int batch = rowIndex / height;
    rowIndex -= height * batch;
    dTensor += batch * batchStride;

    float colsPerRow = width / (float)height;
    float scaledIndex = rowIndex * colsPerRow;
    int baseCol = scaledIndex;
    float shareNextCol = (scaledIndex - baseCol - 1.0f) + colsPerRow;
    shareNextCol *= shareNextCol > 0.0f;
    dTensor[rowIndex * stride + baseCol + (baseCol + 1 != width)] = shareNextCol;
    dTensor[rowIndex * stride + baseCol] = colsPerRow - shareNextCol;
}

void generalIdentityFill(
    int width, int height,
    float *dTensor, int stride, int batchStride,
    int batches
) {
    int threads = 256;
    if (width >= height) {
        int blocks = (width * batches + threads - 1) / threads;
        generalIdentityColumnsKernel<<<blocks, threads>>>(
            width, height,
            dTensor, stride, batchStride,
            batches
        );
    } else {
        int blocks = (height * batches + threads - 1) / threads;
        generalIdentityRowsKernel<<<blocks, threads>>>(
            width, height,
            dTensor, stride, batchStride,
            batches
        );
    }
}

__global__ void positionEmbeddingKernel(
    int biasBits, int temporalBits, int batches, int sequences,
    float* dTensor, int stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batches * sequences * (biasBits + temporalBits)) {
        int y = idx / (biasBits + temporalBits);
        int x = idx - y * (biasBits + temporalBits);
        int sequence = y / batches;
        // int batch = y - batches * sequence;
        int bits = biasBits | (sequence << biasBits);
        dTensor[y * stride + x] = bits >> x & 1;
    }
}

void positionEmbedding(
    int biasBits, int temporalBits, int batches, int sequences,
    float* dTensor, int stride
) {
    assert(biasBits + temporalBits <= 32);
    int blockSize = 256;
    int gridSize = (batches * sequences * (biasBits + temporalBits) + blockSize - 1) / blockSize;
    positionEmbeddingKernel<<<gridSize, blockSize>>>(
        biasBits, temporalBits, batches, sequences,
        dTensor, stride
    );
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

__global__ void softmaxKernel(
    int width,
    const float *dInput, int inputStride,
    float *dOutput, int outputStride,
    int mask
) {
    __shared__ float sTmp[32];
    uint8_t lane = threadIdx.x & 31;
    uint8_t warpId = threadIdx.x >> 5;
    float val = (threadIdx.x < width) ? dInput[blockIdx.x * inputStride + threadIdx.x] : -INFINITY;
    float tmp = val;
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1) tmp = fmaxf(tmp, __shfl_down_sync(0xFFFFFFFF, tmp, i));
    if (lane == 0) sTmp[warpId] = tmp;
    __syncthreads();
    if (warpId == 0) {
        tmp = sTmp[lane];
        #pragma unroll
        for (int i = 16; i > 0; i >>= 1) tmp = fmaxf(tmp, __shfl_down_sync(0xFFFFFFFF, tmp, i));
        if (lane == 0) sTmp[0] = tmp;
    }
    __syncthreads();
    val = expf(val - sTmp[0]);
    tmp = val;
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1) tmp += __shfl_down_sync(0xFFFFFFFF, tmp, i);
    if (lane == 0) sTmp[warpId] = tmp;
    __syncthreads();
    if (warpId == 0) {
        tmp = sTmp[lane];
        #pragma unroll
        for (int i = 16; i > 0; i >>= 1) tmp += __shfl_down_sync(0xFFFFFFFF, tmp, i);
        if (lane == 0) sTmp[0] = tmp;
    }
    __syncthreads();
    if (threadIdx.x < width) dOutput[blockIdx.x * outputStride + threadIdx.x] = val / sTmp[0];
}

void softmax(
    int width, int height,
    const float *dInput, int inputStride,
    float *dOutput, int outputStride,
    int mask
) {
    assert(width <= 1024);
    int blockSize = 1024;
    int gridSize = height;
    softmaxKernel<<<gridSize, blockSize>>>(
        width,
        dInput, inputStride,
        dOutput, outputStride,
        mask
    );
}

__global__ void scoreMaskKernel(
    int batches, int sequences,
    const float *dScores, int scoresSeqStride, int scoresBatchStride,
    float *dOutput, int outputSeqStride, int outputBatchStride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < sequences * batches * sequences) {
        int sequence = idx / (batches * sequences);
        int tmp = idx - sequence * (batches * sequences);
        int batch = tmp / sequences;
        int x = tmp - batch * sequences;
        int scoresIdx = sequence * scoresSeqStride + batch * scoresBatchStride + x;
        int outputIdx = sequence * outputSeqStride + batch * outputBatchStride + x;
        dOutput[outputIdx] = (x > sequence) ? -INFINITY : dScores[scoresIdx];
    }
}

void scoreMask(
    int batches, int sequences,
    const float *dScores, int scoresSeqStride, int scoresBatchStride,
    float *dOutput, int outputSeqStride, int outputBatchStride
) {
    int blockSize = 256;
    int gridSize = (batches * sequences * sequences + blockSize - 1) / blockSize;
    scoreMaskKernel<<<gridSize, blockSize>>>(
        batches, sequences,
        dScores, scoresSeqStride, scoresBatchStride,
        dOutput, outputSeqStride, outputBatchStride
    );
}


__global__ void softmaxGradKernel(
    int width,
    const float* dSoftmax, int softmaxStride,
    const float* dSoftmaxGrad, int softmaxGradStride,
    float* dLogitGrad, int logitStride
) {
    __shared__ float sTmp[32];
    int softmaxIdx = blockIdx.x * softmaxStride + threadIdx.x;
    int gradIdx = blockIdx.x * softmaxGradStride + threadIdx.x;
    int logitIdx = blockIdx.x * logitStride + threadIdx.x;
    uint8_t lane = threadIdx.x & 31;
    uint8_t warpId = threadIdx.x >> 5;
    float dot = (threadIdx.x < width) ? dSoftmax[softmaxIdx] * dSoftmaxGrad[gradIdx] : 0.0f;
    float tmp = dot;
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1) tmp += __shfl_down_sync(0xFFFFFFFF, tmp, i);
    if (lane == 0) sTmp[warpId] = tmp;
    __syncthreads();
    if (warpId == 0) {
        tmp = sTmp[lane];
        #pragma unroll
        for (int i = 16; i > 0; i >>= 1) tmp += __shfl_down_sync(0xFFFFFFFF, tmp, i);
        if (lane == 0) sTmp[0] = tmp;
    }
    __syncthreads();
    if (threadIdx.x < width) dLogitGrad[logitIdx] = dot - dSoftmax[softmaxIdx] * sTmp[0];
}

void softmaxGrad(
    int width, int height,
    const float* dSoftmax, int softmaxStride,
    const float* dSoftmaxGrad, int gradStride,
    float* dLogitGrad, int logitStride
) {
    assert(width <= 1024);
    int blockSize = 1024;
    int gridSize = height;
    softmaxGradKernel<<<gridSize, blockSize>>>(
        width,
        dSoftmax, softmaxStride,
        dSoftmaxGrad, gradStride,
        dLogitGrad, logitStride
    );
}

__global__ void softmaxSampleKernel(
    int width,
    const float *dTensor, int stride,
    float *dOutput, int outputStride,
    int *dSamples,
    int seed
) {
    __shared__ float sTmp[32]; // for softmax computation
    __shared__ float sharedVal[32]; // for partial Gumbel max
    __shared__ int sharedIdx[32]; // for argmax indexing

    int idx = blockIdx.x * stride + threadIdx.x;
    uint8_t lane = threadIdx.x & 31;
    uint8_t warpId = threadIdx.x >> 5;

    // sample softmax logits with Gumbel-max trick using max warp reduce
    float val = (threadIdx.x < width) ? dTensor[idx] : -INFINITY;
    seed += idx * 0xe6546b64;
    seed ^= seed >> 16;
    seed *= 0x85ebca6b;
    seed ^= seed >> 13;
    seed *= 0xc2b2ae35;
    seed ^= seed >> 16;

    // max thread reduce
    float altValGumbel;
    int altIdxGumbel;
    float tmp = val;
    float localMax = val - __logf(-__logf((seed & 0x7fffffff) * 0x1.0p-31 + 1e-8f) + 1e-8f);
    int localIdx = threadIdx.x; // track idx for gumVal
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        tmp = fmaxf(tmp, __shfl_down_sync(0xFFFFFFFF, tmp, offset));
        altValGumbel = __shfl_down_sync(0xffffffff, localMax, offset);
        altIdxGumbel = __shfl_down_sync(0xffffffff, localIdx, offset);
        if (altValGumbel > localMax) {
            localMax = altValGumbel;
            localIdx = altIdxGumbel;
        }
    }
    if (lane == 0) {
        sTmp[warpId] = tmp;
        sharedVal[warpId] = localMax;
        sharedIdx[warpId] = localIdx;
    }
    __syncthreads();
    if (warpId == 0) {
        tmp = sTmp[lane];
        localMax = sharedVal[lane];
        localIdx = sharedIdx[lane];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            tmp = fmaxf(tmp, __shfl_down_sync(0xFFFFFFFF, tmp, offset));
            altValGumbel = __shfl_down_sync(0xffffffff, localMax, offset);
            altIdxGumbel = __shfl_down_sync(0xffffffff, localIdx, offset);
            if (altValGumbel > localMax) {
                localMax = altValGumbel;
                localIdx = altIdxGumbel;
            }
        }
        if (lane == 0) {
            sTmp[0] = tmp;
            dSamples[blockIdx.x] = localIdx;
        }
    }
    __syncthreads();
    val = expf(val - sTmp[0]);
    tmp = val;
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1) tmp += __shfl_down_sync(0xFFFFFFFF, tmp, i);
    if (lane == 0) sTmp[warpId] = tmp;
    __syncthreads();
    if (warpId == 0) {
        tmp = sTmp[lane];
        #pragma unroll
        for (int i = 16; i > 0; i >>= 1) tmp += __shfl_down_sync(0xFFFFFFFF, tmp, i);
        if (lane == 0) sTmp[0] = tmp;
    }
    __syncthreads();
    if (threadIdx.x < width) dOutput[blockIdx.x * outputStride + threadIdx.x] = val / sTmp[0];
}

void softmaxSample(
    int width, int height,
    const float *dTensor, int stride,
    float *dOutput, int outputStride,
    int *dSamples,
    int &seed
) {
    assert(width <= 1024);
    int blockSize = 1024;
    int gridSize = height;
    softmaxSampleKernel<<<gridSize, blockSize>>>(
        width,
        dTensor, stride,
        dOutput, outputStride,
        dSamples,
        seed
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

// THIS IS THE KERNEL THAT IS NOW FIXED
__global__ void residualSwigluGradKernel(
    int width, int height,
    float *dOutputGrad, int outputGradStride, int outputGradBatchStride,
    float *dLogitGrad, int logitGradStride, int logitGradBatchStride,
    float *dResidualGrad, int residualGradStride, int resGradBatchStride, // <-- Renamed
    float *dOutput, int outputStride, int outputBatchStride,
    float *dSigmoid, int sigmoidStride, int sigmoidBatchStride,
    float *dResidual, int residualStride, int resBatchStride, // <-- Renamed
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
    float bias = dResidual[batch * resBatchStride + y * residualStride + x]; // <-- Used new name
    float grad = dOutputGrad[batch * outputGradBatchStride + y * outputGradStride + x];
    dOutputGrad[batch * outputGradBatchStride + y * outputGradStride + x] = grad * sigmoid;
    dLogitGrad[batch * logitGradBatchStride + y * logitGradStride + x] =
        grad * (swiglu - bias) * (1.0f - sigmoid);
    dResidualGrad[batch * resGradBatchStride + y * residualGradStride + x] = grad; // <-- Used new name
}

// THIS IS THE WRAPPER FUNCTION THAT IS NOW FIXED
void residualSwigluGrad(
    int width, int height,
    float *dOutputGrad, int outputGradStride, int outputGradBatchStride,
    float *dLogitGrad, int logitGradStride, int logitGradBatchStride,
    float *dResidualGrad, int residualGradStride, int resGradBatchStride, // <-- Renamed
    float *dOutput, int outputStride, int outputBatchStride,
    float *dSigmoid, int sigmoidStride, int sigmoidBatchStride,
    float *dResidual, int residualStride, int resBatchStride, // <-- Renamed
    int batches
) {
    int blockSize = 256;
    int gridSize = (width * height * batches + blockSize - 1) / blockSize;
    residualSwigluGradKernel<<<gridSize, blockSize>>>(
        width, height,
        dOutputGrad, outputGradStride, outputGradBatchStride,
        dLogitGrad, logitGradStride, logitGradBatchStride,
        dResidualGrad, residualGradStride, resGradBatchStride, // <-- Pass new name
        dOutput, outputStride, outputBatchStride,
        dSigmoid, sigmoidStride, sigmoidBatchStride,
        dResidual, residualStride, resBatchStride, // <-- Pass new name
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

__global__ void swap2DKernel(
    int width, int height,
    float* dTensorA, int strideA,
    float* dTensorB, int strideB
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    int y = idx / width;
    int x = idx - y * width;

    int indexA = y * strideA + x;
    int indexB = y * strideB + x;

    float tmp = dTensorA[indexA];
    dTensorA[indexA] = dTensorB[indexB];
    dTensorB[indexB] = tmp;
}

void swap2D(
    int width, int height,
    float* dTensorA, int strideA,
    float* dTensorB, int strideB
) {
    int blockSize = 256;
    int gridSize = (width * height + blockSize - 1) / blockSize;
    swap2DKernel<<<gridSize, blockSize>>>(
        width, height,
        dTensorA, strideA,
        dTensorB, strideB
    );
}

__global__ void gaeGradientKernel(
    int batches, int sequences, int actions,
    const float* dValue, int valueStride,
    const float* dProb,
    const float* dReward,
    const int* dSamples,
    float* dOutputGrad, int outputGradStride,
    float DISCOUNT, float LAMBDA, float entropy_beta
) {
    __shared__ float sTmp[64];
    __shared__ float sTmp2[64];
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    uint8_t warpTid = tid & 31;
    uint8_t warp = tid >> 5;
    float discount = DISCOUNT;
    float reward = tid < sequences ? dReward[tid * batches + bid] : 0.0f;
    float lamda = DISCOUNT * LAMBDA;
    float val = tid < sequences ? dValue[tid * batches * valueStride + bid * valueStride] : 0.0f;
    float valNext = tid < sequences - 1 ? dValue[(tid + 1) * batches * valueStride + bid * valueStride] : 0.0f;
    float advantage = tid < sequences ? reward + discount * valNext - val : 0.0f;
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1) {
        advantage += (warpTid < 32 - i) * lamda * __shfl_down_sync(0xFFFFFFFF, advantage, i);
        lamda *= (warpTid < 32 - i) * (__shfl_down_sync(0xffffffff, lamda, i) - 1) + 1;
    }
    if (warpTid == 0) {
        sTmp2[warp] = advantage;
        sTmp2[warp + 32] = lamda;
    }
    __syncthreads();
    if (warp == 0) {
        float sum = sTmp[warpTid];
        float dis = sTmp[warpTid + 32];
        float gAdvantage = sTmp2[warpTid];
        float gLamda = sTmp2[warpTid + 32];
        #pragma unroll
        for (int i = 1; i < 32; i <<= 1) {
            sum += (warpTid < 32 - i) * dis * __shfl_down_sync(0xFFFFFFFF, sum, i);
            dis *= (warpTid < 32 - i) * (__shfl_down_sync(0xffffffff, dis, i) - 1) + 1;
            gAdvantage += (warpTid < 32 - i) * gLamda * __shfl_down_sync(0xFFFFFFFF, gAdvantage, i);
            gLamda *= (warpTid < 32 - i) * (__shfl_down_sync(0xffffffff, gLamda, i) - 1) + 1;
        }
        sTmp[warpTid] = sum;
        sTmp2[warpTid] = gAdvantage;
    }
    __syncthreads();
    advantage += warp != 31 ? lamda * sTmp2[warp + 1] : 0.0f;
    if (tid < sequences) {
        float entropy = 0.0f;
        for (int i = 0; i < actions; i++) {
            float prob = dProb[tid * batches * actions + bid * actions + i];
            entropy -= prob * logf(prob + 1e-8f);
        }
        dOutputGrad[tid * batches * outputGradStride + bid * outputGradStride + actions] = 1.0f * (reward + discount * valNext - val);
        for (int i = 0; i < actions; i++) {
            float prob = dProb[tid * batches * actions + bid * actions + i];
            dOutputGrad[tid * batches * outputGradStride + bid * outputGradStride + i] = 1 * advantage * ((i == dSamples[tid * batches + bid]) - prob) + entropy_beta * (prob * (entropy + logf(prob + 1e-8f)));
        }
    }
}

void gaeGradient(
    int batches, int sequences, int actions,
    const float* dValue, int valueStride,
    const float* dProb,
    const float* dReward,
    const int* dSamples,
    float* dOutputGrad, int outputGradStride,
    float DISCOUNT, float LAMBDA, float entropy_beta
) {
    assert(sequences <= 1024);
    int blockSize = 1024;
    int gridSize = batches;
    gaeGradientKernel<<<gridSize, blockSize>>>(
        batches, sequences, actions,
        dValue, valueStride,
        dProb,
        dReward,
        dSamples,
        dOutputGrad, outputGradStride,
        DISCOUNT, LAMBDA, entropy_beta
    );
}
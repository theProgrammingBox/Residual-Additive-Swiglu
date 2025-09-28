#include "header.cuh"

#define LOG_SKIPS 1
#define EPOCHS 1
#define BATCH_ITERATIONS 1
#define BATCH_SIZE 3

#define INPUT_BITS 2
#define RESIDUAL_SIZE 6
#define SWIGLUS 1
#define LAYERS 1

int main() {
    assert(INPUT_BITS > 0 && INPUT_BITS <= 32);
    
    int seed = 0;
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    const float ONE = 1.0f;
    const float ZERO = 0.0f;
    const float NEG_ONE = -1.0f;
    
    float *hInput = (float*)malloc((INPUT_BITS * 2) * BATCH_SIZE * sizeof(float));
    float *hOutput = (float*)malloc((INPUT_BITS * 2) * BATCH_SIZE * sizeof(float));
    
    float* dForward = callocDeviceTensor<float>(RESIDUAL_SIZE * SWIGLUS * 2 * BATCH_SIZE * (LAYERS + 1));
    float* dNorm = mallocDeviceTensor<float>(RESIDUAL_SIZE * BATCH_SIZE);
    
    float* dBackwardTop = mallocDeviceTensor<float>(RESIDUAL_SIZE * SWIGLUS * 2 * BATCH_SIZE);
    float* dBackwardBottom = mallocDeviceTensor<float>(RESIDUAL_SIZE * SWIGLUS * 2 * BATCH_SIZE);
    float* dBackwardTmp;
    
    float* dSwigluSumWeights = mallocDeviceTensor<float>(RESIDUAL_SIZE * SWIGLUS * 2 * RESIDUAL_SIZE * LAYERS);
    float* dSwigluSumWeightGrads = mallocDeviceTensor<float>(RESIDUAL_SIZE * SWIGLUS * 2 * RESIDUAL_SIZE * LAYERS);
    float* dSwigluSumWeightGradMeans = callocDeviceTensor<float>(RESIDUAL_SIZE * SWIGLUS * 2 * RESIDUAL_SIZE * LAYERS);
    float* dSwigluSumWeightGradVars = callocDeviceTensor<float>(RESIDUAL_SIZE * SWIGLUS * 2 * RESIDUAL_SIZE * LAYERS);
    normalRandFill(
        RESIDUAL_SIZE * SWIGLUS * 2, RESIDUAL_SIZE * LAYERS,
        dSwigluSumWeights, RESIDUAL_SIZE * SWIGLUS * 2,
        seed, 0.0f, 0.02f
    );
    
    time_t start_time, end_time;
    time(&start_time);
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        // reset gradients
        cudaMemset(dSwigluSumWeightGrads, 0, RESIDUAL_SIZE * SWIGLUS * 2 * RESIDUAL_SIZE * LAYERS * sizeof(float));
        
        float totalLoss = 0.0f;
        for (int batchIteration = 0; batchIteration < BATCH_ITERATIONS; batchIteration++) {
            // Fill input data
            unsigned int mask = (1u << INPUT_BITS) - 1u;
            for (int batch = 0; batch < BATCH_SIZE; batch++) {
                unsigned int a = (unsigned int)hashI32(seed) & mask;
                unsigned int b = (unsigned int)hashI32(seed) & mask;
                int offset = batch * (INPUT_BITS * 2);
                for (int bit = 0; bit < INPUT_BITS; bit++) {
                    hInput[offset + bit] = (float)((a >> bit) & 1u);
                    hInput[offset + INPUT_BITS + bit] = (float)((b >> bit) & 1u);
                }
                unsigned int sum = a + b;
                for (int bit = 0; bit < INPUT_BITS * 2; bit++) {
                    hOutput[offset + bit] = (float)((sum >> bit) & 1u);
                }
            }
            
            // copy input to device
            cudaMemcpy2D(
                dForward, RESIDUAL_SIZE * SWIGLUS * 2 * sizeof(float),
                hInput, INPUT_BITS * 2 * sizeof(float),
                INPUT_BITS * 2 * sizeof(float), BATCH_SIZE,
                cudaMemcpyHostToDevice
            );
            printDeviceTensor(
                "input, only first %d used",
                RESIDUAL_SIZE * SWIGLUS * 2, BATCH_SIZE,
                dForward, RESIDUAL_SIZE * SWIGLUS * 2, INPUT_BITS * 2
            );
            
            // copy output to device
            cudaMemcpy2D(
                dBackwardTop, RESIDUAL_SIZE * SWIGLUS * 2 * sizeof(float),
                hOutput, INPUT_BITS * 2 * sizeof(float),
                INPUT_BITS * 2 * sizeof(float), BATCH_SIZE,
                cudaMemcpyHostToDevice
            );
            printDeviceTensor(
                "target, only first %d used",
                RESIDUAL_SIZE * SWIGLUS * 2, BATCH_SIZE,
                dBackwardTop, RESIDUAL_SIZE * SWIGLUS * 2, INPUT_BITS * 2
            );
            
            // layered forward pass
            for (int layer = 0; layer < LAYERS; layer++) {
                int forwardLayerOffset = RESIDUAL_SIZE * SWIGLUS * 2 * BATCH_SIZE * layer;
                int nextForwardLayerOffset = RESIDUAL_SIZE * SWIGLUS * 2 * BATCH_SIZE * (layer + 1);
                int swigluSumWeightLayerOffset = RESIDUAL_SIZE * SWIGLUS * 2 * RESIDUAL_SIZE * layer;
                
                // batchnorm
                batchNorm(
                    RESIDUAL_SIZE, BATCH_SIZE,
                    dForward + forwardLayerOffset, RESIDUAL_SIZE * SWIGLUS * 2, 0,
                    dNorm, RESIDUAL_SIZE, 0,
                    1
                );
                printDeviceTensor(
                    "norm %d",
                    RESIDUAL_SIZE, BATCH_SIZE,
                    dNorm, RESIDUAL_SIZE, layer
                );
                
                // swiglus gemm
                cublasGemmEx(
                    cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                    RESIDUAL_SIZE * SWIGLUS * 2, BATCH_SIZE, RESIDUAL_SIZE,
                    &ONE,
                    dSwigluSumWeights + swigluSumWeightLayerOffset, CUDA_R_32F, RESIDUAL_SIZE * SWIGLUS * 2,
                    dNorm, CUDA_R_32F, RESIDUAL_SIZE,
                    &ZERO,
                    dForward + nextForwardLayerOffset, CUDA_R_32F, RESIDUAL_SIZE * SWIGLUS * 2,
                    CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT
                );
                printDeviceTensor(
                    "swiglu gemm %d",
                    RESIDUAL_SIZE * SWIGLUS * 2, BATCH_SIZE,
                    dForward + nextForwardLayerOffset, RESIDUAL_SIZE * SWIGLUS * 2, layer + 1
                );
                
                // residual swiglu sum
                // TODO error in offset for swiglu
                residualSwiglu(
                    RESIDUAL_SIZE, BATCH_SIZE,
                    dForward + nextForwardLayerOffset, RESIDUAL_SIZE * SWIGLUS * 2, 0,
                    dForward + nextForwardLayerOffset + RESIDUAL_SIZE, RESIDUAL_SIZE * SWIGLUS * 2, 0,
                    dForward + forwardLayerOffset, RESIDUAL_SIZE * SWIGLUS * 2, 0,
                    1
                );
                printDeviceTensor(
                    "residual swiglu %d",
                    RESIDUAL_SIZE * SWIGLUS * 2, BATCH_SIZE,
                    dForward + nextForwardLayerOffset, RESIDUAL_SIZE * SWIGLUS * 2, layer + 1
                );
            }
            
            // get error and clear unused part of backward top
            cublasSgeam(
                cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                INPUT_BITS * 2, BATCH_SIZE,
                &NEG_ONE,
                dForward + RESIDUAL_SIZE * SWIGLUS * 2 * BATCH_SIZE * LAYERS, RESIDUAL_SIZE * SWIGLUS * 2,
                &ONE,
                dBackwardTop, RESIDUAL_SIZE * SWIGLUS * 2,
                dBackwardTop, RESIDUAL_SIZE * SWIGLUS * 2
            );
            cudaMemset2D(
                dBackwardTop + INPUT_BITS * 2, RESIDUAL_SIZE * SWIGLUS * 2 * sizeof(float),
                0, (RESIDUAL_SIZE * SWIGLUS * 2 - INPUT_BITS * 2) * sizeof(float), BATCH_SIZE
            );
            printDeviceTensor(
                "error, only first %d used",
                RESIDUAL_SIZE * SWIGLUS * 2, BATCH_SIZE,
                dBackwardTop, RESIDUAL_SIZE * SWIGLUS * 2, INPUT_BITS * 2
            );
            
            // accumulate loss
            if ((epoch + 1) % LOG_SKIPS == 0) {
                float batchLoss;
                cublasSdot(
                    cublasHandle, RESIDUAL_SIZE * SWIGLUS * 2 * BATCH_SIZE,
                    dBackwardTop, 1, dBackwardTop, 1, &batchLoss
                );
                totalLoss += batchLoss;
            }
            
            // layered backward pass
            for (int layer = LAYERS; layer--;) {
                int forwardLayerOffset = RESIDUAL_SIZE * SWIGLUS * 2 * BATCH_SIZE * layer;
                int nextForwardLayerOffset = RESIDUAL_SIZE * SWIGLUS * 2 * BATCH_SIZE * (layer + 1);
                int swigluSumWeightLayerOffset = RESIDUAL_SIZE * SWIGLUS * 2 * RESIDUAL_SIZE * layer;
                
                // residual swiglu grad
                residualSwigluGrad(
                    RESIDUAL_SIZE, BATCHES,
                    dBackwardTop, 2 * RESIDUAL_SIZE, BATCHES * (2 * RESIDUAL_SIZE),
                    dBackwardTop + RESIDUAL_SIZE, 2 * RESIDUAL_SIZE, BATCHES * (2 * RESIDUAL_SIZE),
                    dBackwardBottom, 2 * RESIDUAL_SIZE, BATCHES * (2 * RESIDUAL_SIZE),
                    dForward[layer + 1], 2 * RESIDUAL_SIZE, BATCHES * (2 * RESIDUAL_SIZE),
                    dForward[layer + 1] + RESIDUAL_SIZE, 2 * RESIDUAL_SIZE, BATCHES * (2 * RESIDUAL_SIZE),
                    dForward[layer], 2 * RESIDUAL_SIZE, BATCHES * (2 * RESIDUAL_SIZE),
                    1
                );
            }
        }
        
        // log loss
        if ((epoch + 1) % LOG_SKIPS == 0) {
            printf("Epoch %u: %.6f\n", epoch + 1, 0.5f * totalLoss / (BATCH_SIZE * BATCH_ITERATIONS));
        }
        
        // apply gradients
    }
    time(&end_time);
    printf("Simulation time: %ld seconds\n", end_time - start_time);
    
    return 0;
}


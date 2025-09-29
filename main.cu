#include "header.cuh"

#define LOG_SKIPS (64 * 1)
#define EPOCHS (1024 * 1)
#define BATCH_ITERATIONS (1)
#define BATCH_SIZE (1024)

#define INPUT_BITS (16)
#define RESIDUAL_SIZE (256)
#define SWIGLUS (1)
#define LAYERS (32 * 16)

#define K 0.0001f
#define LR 0.001f
#define MEAN_BETA 0.9f
#define VAR_BETA 0.999f
#define EPSILON 1e-8f

int main() {
    assert(INPUT_BITS > 0 && INPUT_BITS <= 16);
    
    int seed = 0;
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    const float ONE = 1.0f;
    const float ZERO = 0.0f;
    const float NEG_ONE = -1.0f;
    const float LR_SCALE = 1.0f / (INPUT_BITS * 2 * BATCH_SIZE * BATCH_ITERATIONS);
    
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
        // seed, 0.0f, 0.02f
        seed, 0.0f, rsqrtf(RESIDUAL_SIZE)
    );
    
    time_t start_time, end_time;
    time(&start_time);
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        // reset gradients
        cudaMemset(dSwigluSumWeightGrads, 0, RESIDUAL_SIZE * SWIGLUS * 2 * RESIDUAL_SIZE * LAYERS * sizeof(float));
        
        float totalLoss = 0.0f;
        float totalError = 0.0f;
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
                unsigned int c = a + b;
                // unsigned int c = b << INPUT_BITS | a;
                for (int bit = 0; bit < INPUT_BITS * 2; bit++) {
                    hOutput[offset + bit] = (float)((c >> bit) & 1u);
                }
            }
            
            // copy input to device
            cudaMemcpy2D(
                dForward, RESIDUAL_SIZE * SWIGLUS * 2 * sizeof(float),
                hInput, INPUT_BITS * 2 * sizeof(float),
                INPUT_BITS * 2 * sizeof(float), BATCH_SIZE,
                cudaMemcpyHostToDevice
            );
            // printDeviceTensor(
            //     "input, only first %d used",
            //     RESIDUAL_SIZE * SWIGLUS * 2, BATCH_SIZE,
            //     dForward, RESIDUAL_SIZE * SWIGLUS * 2, INPUT_BITS * 2
            // );
            
            // copy output to device
            cudaMemcpy2D(
                dBackwardTop, RESIDUAL_SIZE * SWIGLUS * 2 * sizeof(float),
                hOutput, INPUT_BITS * 2 * sizeof(float),
                INPUT_BITS * 2 * sizeof(float), BATCH_SIZE,
                cudaMemcpyHostToDevice
            );
            // printDeviceTensor(
            //     "target, only first %d used",
            //     RESIDUAL_SIZE * SWIGLUS * 2, BATCH_SIZE,
            //     dBackwardTop, RESIDUAL_SIZE * SWIGLUS * 2, INPUT_BITS * 2
            // );
            
            // layered forward pass
            for (int layer = 0; layer < LAYERS; layer++) {
                int forwardLayerOffset = RESIDUAL_SIZE * SWIGLUS * 2 * BATCH_SIZE * layer;
                int nextForwardLayerOffset = RESIDUAL_SIZE * SWIGLUS * 2 * BATCH_SIZE + forwardLayerOffset;
                int swigluSumWeightLayerOffset = RESIDUAL_SIZE * SWIGLUS * 2 * RESIDUAL_SIZE * layer;
                
                // batchnorm
                batchNorm(
                    RESIDUAL_SIZE, BATCH_SIZE,
                    dForward + forwardLayerOffset, RESIDUAL_SIZE * SWIGLUS * 2, 0,
                    dNorm, RESIDUAL_SIZE, 0,
                    1
                );
                // printDeviceTensor(
                //     "norm %d",
                //     RESIDUAL_SIZE, BATCH_SIZE,
                //     dNorm, RESIDUAL_SIZE, layer
                // );
                
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
                // printDeviceTensor(
                //     "swiglu gemm %d",
                //     RESIDUAL_SIZE * SWIGLUS * 2, BATCH_SIZE,
                //     dForward + nextForwardLayerOffset, RESIDUAL_SIZE * SWIGLUS * 2, layer + 1
                // );
                
                // residual swiglu sum
                residualSwigluSum(
                    RESIDUAL_SIZE, BATCH_SIZE, SWIGLUS,
                    dForward + nextForwardLayerOffset, RESIDUAL_SIZE * SWIGLUS * 2, 0, RESIDUAL_SIZE,
                    dForward + nextForwardLayerOffset + RESIDUAL_SIZE * SWIGLUS, RESIDUAL_SIZE * SWIGLUS * 2, 0, RESIDUAL_SIZE,
                    dForward + forwardLayerOffset, RESIDUAL_SIZE * SWIGLUS * 2, 0,
                    1
                );
                // residualSwiglu(
                //     RESIDUAL_SIZE, BATCH_SIZE,
                //     dForward + nextForwardLayerOffset, RESIDUAL_SIZE * SWIGLUS * 2, 0,
                //     dForward + nextForwardLayerOffset + RESIDUAL_SIZE * SWIGLUS, RESIDUAL_SIZE * SWIGLUS * 2, 0,
                //     dForward + forwardLayerOffset, RESIDUAL_SIZE * SWIGLUS * 2, 0,
                //     1
                // );
                // printDeviceTensor(
                //     "residual swiglu %d",
                //     RESIDUAL_SIZE * SWIGLUS * 2, BATCH_SIZE,
                //     dForward + nextForwardLayerOffset, RESIDUAL_SIZE * SWIGLUS * 2, layer + 1
                // );
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
            // printDeviceTensor(
            //     "error, only first %d used",
            //     RESIDUAL_SIZE * SWIGLUS * 2, BATCH_SIZE,
            //     dBackwardTop, RESIDUAL_SIZE * SWIGLUS * 2, INPUT_BITS * 2
            // );
            
            // accumulate loss
            if ((epoch + 1) % LOG_SKIPS == 0) {
                float batchLoss = 0.0f;
                float batchError = 0.0f;
                cublasSdot(
                    cublasHandle, RESIDUAL_SIZE * SWIGLUS * 2 * BATCH_SIZE,
                    dBackwardTop, 1, dBackwardTop, 1, &batchLoss
                );
                // cublasSasum(
                //     cublasHandle, RESIDUAL_SIZE * SWIGLUS * 2 * BATCH_SIZE,
                //     dBackwardTop, 1, &batchError
                // );
                cublasNrm2Ex(
                    cublasHandle, RESIDUAL_SIZE * SWIGLUS * 2 * BATCH_SIZE,
                    dBackwardTop, CUDA_R_32F, 1,
                    &batchError, CUDA_R_32F, CUDA_R_32F
                );
                totalLoss += batchLoss;
                totalError += batchError;
            }
            
            // layered backward pass
            for (int layer = LAYERS; layer--;) {
                int forwardLayerOffset = RESIDUAL_SIZE * SWIGLUS * 2 * BATCH_SIZE * layer;
                int nextForwardLayerOffset = RESIDUAL_SIZE * SWIGLUS * 2 * BATCH_SIZE + forwardLayerOffset;
                int swigluSumWeightLayerOffset = RESIDUAL_SIZE * SWIGLUS * 2 * RESIDUAL_SIZE * layer;
                
                // residual swiglu grad
                residualSwigluSumGrad(
                    RESIDUAL_SIZE, BATCH_SIZE, SWIGLUS,
                    dBackwardTop, RESIDUAL_SIZE * SWIGLUS * 2, 0, RESIDUAL_SIZE,
                    dBackwardTop + RESIDUAL_SIZE * SWIGLUS, RESIDUAL_SIZE * SWIGLUS * 2, 0, RESIDUAL_SIZE,
                    dBackwardBottom, RESIDUAL_SIZE * SWIGLUS * 2, 0,
                    dForward + nextForwardLayerOffset, RESIDUAL_SIZE * SWIGLUS * 2, 0, RESIDUAL_SIZE,
                    dForward + nextForwardLayerOffset + RESIDUAL_SIZE * SWIGLUS, RESIDUAL_SIZE * SWIGLUS * 2, 0, RESIDUAL_SIZE,
                    dForward + forwardLayerOffset, RESIDUAL_SIZE * SWIGLUS * 2, 0,
                    1
                );
                // residualSwigluGrad(
                //     RESIDUAL_SIZE, BATCH_SIZE,
                //     dBackwardTop, RESIDUAL_SIZE * SWIGLUS * 2, 0,
                //     dBackwardTop + RESIDUAL_SIZE * SWIGLUS, RESIDUAL_SIZE * SWIGLUS * 2, 0,
                //     dBackwardBottom, RESIDUAL_SIZE * SWIGLUS * 2, 0,
                //     dForward + nextForwardLayerOffset, RESIDUAL_SIZE * SWIGLUS * 2, 0,
                //     dForward + nextForwardLayerOffset + RESIDUAL_SIZE * SWIGLUS, RESIDUAL_SIZE * SWIGLUS * 2, 0,
                //     dForward + forwardLayerOffset, RESIDUAL_SIZE * SWIGLUS * 2, 0,
                //     1
                // );
                // printDeviceTensor(
                //     "residual swiglu grad %d",
                //     RESIDUAL_SIZE * SWIGLUS * 2, BATCH_SIZE,
                //     dBackwardTop, RESIDUAL_SIZE * SWIGLUS * 2, layer + 1
                // );
                
                // swiglu gemm grad
                cublasGemmEx(
                    cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                    RESIDUAL_SIZE, BATCH_SIZE, RESIDUAL_SIZE * SWIGLUS * 2,
                    &ONE,
                    dSwigluSumWeights + swigluSumWeightLayerOffset, CUDA_R_32F, RESIDUAL_SIZE * SWIGLUS * 2,
                    dBackwardTop, CUDA_R_32F, RESIDUAL_SIZE * SWIGLUS * 2,
                    &ZERO,
                    dNorm, CUDA_R_32F, RESIDUAL_SIZE,
                    CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT
                );
                // printDeviceTensor(
                //     "swiglu gemm grad norm %d",
                //     RESIDUAL_SIZE, BATCH_SIZE,
                //     dNorm, RESIDUAL_SIZE, layer
                // );
                
                // batchnorm grad
                batchNormGrad(
                    RESIDUAL_SIZE, BATCH_SIZE,
                    dNorm, RESIDUAL_SIZE, 0,
                    dForward + forwardLayerOffset, RESIDUAL_SIZE * SWIGLUS * 2, 0,
                    dBackwardBottom, RESIDUAL_SIZE * SWIGLUS * 2, 0,
                    1
                );
                // printDeviceTensor(
                //     "residual batchnorm grad %d",
                //     RESIDUAL_SIZE * SWIGLUS * 2, BATCH_SIZE,
                //     dBackwardBottom, RESIDUAL_SIZE * SWIGLUS * 2, layer
                // );
                
                // swiglu gemm weight grad
                cublasGemmEx(
                    cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                    RESIDUAL_SIZE * SWIGLUS * 2, RESIDUAL_SIZE, BATCH_SIZE,
                    // &ONE,
                    &LR_SCALE,
                    dBackwardTop, CUDA_R_32F, RESIDUAL_SIZE * SWIGLUS * 2,
                    dNorm, CUDA_R_32F, RESIDUAL_SIZE,
                    &ONE,
                    dSwigluSumWeightGrads + swigluSumWeightLayerOffset, CUDA_R_32F, RESIDUAL_SIZE * SWIGLUS * 2,
                    CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT
                );
                // printDeviceTensor(
                //     "swiglu gemm weight grad %d",
                //     RESIDUAL_SIZE * SWIGLUS * 2, RESIDUAL_SIZE,
                //     dSwigluSumWeightGrads + swigluSumWeightLayerOffset, RESIDUAL_SIZE * SWIGLUS * 2, layer
                // );
                
                // swap backward buffers
                dBackwardTmp = dBackwardTop;
                dBackwardTop = dBackwardBottom;
                dBackwardBottom = dBackwardTmp;
            }
        }
        
        // log loss
        if ((epoch + 1) % LOG_SKIPS == 0) {
            // printf("Epoch %u: loss %.6f, error %.6f\n", epoch + 1, 0.5f * totalLoss * LR_SCALE, totalError * LR_SCALE);
            printf("Epoch %u: loss %.6f, error %.6f\n", epoch + 1, 0.5f * totalLoss * LR_SCALE, totalError * rsqrtf(BATCH_SIZE * INPUT_BITS * 2));
        }
        
        // apply gradients
        adamUpdate(
            RESIDUAL_SIZE * SWIGLUS * 2, RESIDUAL_SIZE * LAYERS,
            dSwigluSumWeightGrads, RESIDUAL_SIZE * SWIGLUS * 2,
            dSwigluSumWeightGradMeans, RESIDUAL_SIZE * SWIGLUS * 2,
            dSwigluSumWeightGradVars, RESIDUAL_SIZE * SWIGLUS * 2,
            dSwigluSumWeights, RESIDUAL_SIZE * SWIGLUS * 2,
            LR, MEAN_BETA, VAR_BETA, EPSILON
        );
        // printDeviceTensor(
        //     "swiglu weights",
        //     RESIDUAL_SIZE * SWIGLUS * 2, RESIDUAL_SIZE * LAYERS,
        //     dSwigluSumWeights, RESIDUAL_SIZE * SWIGLUS * 2
        // );
    }
    time(&end_time);
    printf("Simulation time: %ld seconds\n", end_time - start_time);
    
    return 0;
}
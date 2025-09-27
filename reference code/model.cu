#include "model.hpp"
#include "header.cuh"   // <-- compile kernels & helpers exactly once here

Model::Model(int& seed, cublasHandle_t handle)
: dActions(nullptr), dProbs(nullptr), dReward(nullptr),
  dForward(nullptr), dNorm(nullptr), dResult(nullptr), dQKV(nullptr), dScore(nullptr),
  dBackwardTop(nullptr), dBackwardBottom(nullptr), dQKVGrad(nullptr), dScoreGrad(nullptr),
  dQKVWeights(nullptr), dQKVWeightGrads(nullptr), dQKVWeightGradMeans(nullptr), dQKVWeightGradVars(nullptr),
  dSwigluWeights(nullptr), dSwigluWeightGrads(nullptr), dSwigluWeightGradMeans(nullptr), dSwigluWeightGradVars(nullptr),
  averageReward(0.0f), pSeed(&seed), cublasHandle(handle),
  ONE(1.0f), ZERO(0.0f),
  ROOT_QUERY_D(rsqrtf(QUERY_D)),
  LR_NORM(rsqrtf(BATCHES * BATCH_ITERATIONS * SEQUENCES))
{
    // Device allocations (same as original)
    dActions = mallocDeviceTensor<int>(BATCHES * SEQUENCES);
    dProbs   = mallocDeviceTensor<float>(ACTIONS * BATCHES * SEQUENCES);
    dReward  = mallocDeviceTensor<float>(BATCHES * SEQUENCES);

    dForward = mallocDeviceTensor<float>(CONCAT_SWIGLU_D * BATCHES * SEQUENCES * (LAYERS + 1));
    dNorm    = mallocDeviceTensor<float>(CONCAT_NORM_D   * BATCHES * SEQUENCES);
    dResult  = mallocDeviceTensor<float>(VALUE_D * HEADS * BATCHES * SEQUENCES);
    dQKV     = mallocDeviceTensor<float>(QKV_D * HEADS   * BATCHES * SEQUENCES * LAYERS);
    dScore   = mallocDeviceTensor<float>(SEQUENCES * SEQUENCES * HEADS * BATCHES);

    dBackwardTop    = mallocDeviceTensor<float>(CONCAT_SWIGLU_D * BATCHES * SEQUENCES);
    dBackwardBottom = mallocDeviceTensor<float>(CONCAT_SWIGLU_D * BATCHES * SEQUENCES);
    dQKVGrad        = mallocDeviceTensor<float>(QKV_D * HEADS * BATCHES * SEQUENCES);
    dScoreGrad      = mallocDeviceTensor<float>(SEQUENCES * SEQUENCES * HEADS * BATCHES);

    dQKVWeights          = callocDeviceTensor<float>(QKV_D * HEADS * LAYERS * LATENT_D);
    dQKVWeightGrads      = mallocDeviceTensor<float>(QKV_D * HEADS * LAYERS * LATENT_D);
    dQKVWeightGradMeans  = callocDeviceTensor<float>(QKV_D * HEADS * LAYERS * LATENT_D);
    dQKVWeightGradVars   = callocDeviceTensor<float>(QKV_D * HEADS * LAYERS * LATENT_D);

    dSwigluWeights       = mallocDeviceTensor<float>(CONCAT_SWIGLU_D * CONCAT_NORM_D * LAYERS);
    dSwigluWeightGrads   = mallocDeviceTensor<float>(CONCAT_SWIGLU_D * CONCAT_NORM_D * LAYERS);
    dSwigluWeightGradMeans = callocDeviceTensor<float>(CONCAT_SWIGLU_D * CONCAT_NORM_D * LAYERS);
    dSwigluWeightGradVars  = callocDeviceTensor<float>(CONCAT_SWIGLU_D * CONCAT_NORM_D * LAYERS);
}

Model::~Model() {
    // Free device memory
    cudaFree(dActions);
    cudaFree(dProbs);
    cudaFree(dReward);

    cudaFree(dForward);
    cudaFree(dNorm);
    cudaFree(dResult);
    cudaFree(dQKV);
    cudaFree(dScore);

    cudaFree(dBackwardTop);
    cudaFree(dBackwardBottom);
    cudaFree(dQKVGrad);
    cudaFree(dScoreGrad);

    cudaFree(dQKVWeights);
    cudaFree(dQKVWeightGrads);
    cudaFree(dQKVWeightGradMeans);
    cudaFree(dQKVWeightGradVars);

    cudaFree(dSwigluWeights);
    cudaFree(dSwigluWeightGrads);
    cudaFree(dSwigluWeightGradMeans);
    cudaFree(dSwigluWeightGradVars);
}

void Model::initWeights(bool load_from_files) {
    if (load_from_files) {
        loadDeviceTensor("qkv_weights.bin", dQKVWeights, QKV_D * HEADS * LAYERS * LATENT_D);
        loadDeviceTensor("qkv_weight_grad_means.bin", dQKVWeightGradMeans, QKV_D * HEADS * LAYERS * LATENT_D);
        loadDeviceTensor("qkv_weight_grad_vars.bin",  dQKVWeightGradVars,  QKV_D * HEADS * LAYERS * LATENT_D);
        loadDeviceTensor("swiglu_weights.bin", dSwigluWeights, CONCAT_SWIGLU_D * CONCAT_NORM_D * LAYERS);
        loadDeviceTensor("swiglu_weight_grad_means.bin", dSwigluWeightGradMeans, CONCAT_SWIGLU_D * CONCAT_NORM_D * LAYERS);
        loadDeviceTensor("swiglu_weight_grad_vars.bin",  dSwigluWeightGradVars,  CONCAT_SWIGLU_D * CONCAT_NORM_D * LAYERS);
    } else {
        generalIdentityFill(
            QUERY_D, LATENT_D,
            dQKVWeights, QKV_D * HEADS * LAYERS, QKV_D,
            HEADS * LAYERS
        );
        generalIdentityFill(
            QUERY_D, LATENT_D,
            dQKVWeights + QUERY_D, QKV_D * HEADS * LAYERS, QKV_D,
            HEADS * LAYERS
        );
        generalIdentityFill(
            VALUE_D, LATENT_D,
            dQKVWeights + 2 * QUERY_D, QKV_D * HEADS * LAYERS, QKV_D,
            HEADS * LAYERS
        );

        normalRandFill(
            CONCAT_SWIGLU_D * CONCAT_NORM_D * LAYERS, 1,
            dSwigluWeights, CONCAT_SWIGLU_D,
            *pSeed, 0, rsqrtf(CONCAT_NORM_D)
        );
    }
}

void Model::positionEmbedding() {
    ::positionEmbedding(  // qualify to call the global function, not the method
        BIAS_BITS, TEMPORAL_BITS, BATCHES, SEQUENCES,
        dForward + LATENT_D - (BIAS_BITS + TEMPORAL_BITS), CONCAT_SWIGLU_D
    );
}

void Model::zeroGradsAndResetLogging() {
    cudaMemset(dQKVWeightGrads,    0, QKV_D * HEADS * LAYERS * LATENT_D * sizeof(float));
    cudaMemset(dSwigluWeightGrads, 0, CONCAT_SWIGLU_D * CONCAT_NORM_D * LAYERS * sizeof(float));
    averageReward = 0.0f;
}

void Model::forwardSequence(int sequence, const Environment& env) {
    const int forwardSeqOff = forwardSeqOffset(sequence);
    const int QKVSeqOffset  = QKV_D * HEADS * BATCHES * sequence;
    const int scoreSeqOff   = SEQUENCES * sequence;

    cudaMemcpy2D(
        dForward + forwardSeqOff, CONCAT_SWIGLU_D * sizeof(float),
        env.board(), BOARD_AREA * sizeof(float),
        BOARD_AREA * sizeof(float), BATCHES, cudaMemcpyHostToDevice
    );

    for (int layer = 0; layer < LAYERS; layer++) {
        const int forwardLayerOff     = CONCAT_SWIGLU_D * BATCHES * SEQUENCES * layer;
        const int nextForwardLayerOff = CONCAT_SWIGLU_D * BATCHES * SEQUENCES * (layer + 1);
        const int QKVWeightLayerOff   = QKV_D * HEADS * layer;
        const int QKVLayerOff         = QKV_D * HEADS * BATCHES * SEQUENCES * layer;
        const int swigluWeightLayerOff= CONCAT_SWIGLU_D * CONCAT_NORM_D * layer;

        batchNorm(
            LATENT_D, BATCHES,
            dForward + forwardSeqOff + forwardLayerOff, CONCAT_SWIGLU_D, CONCAT_SWIGLU_D * BATCHES,
            dNorm,    CONCAT_NORM_D, CONCAT_NORM_D * BATCHES,
            1
        );

        cublasGemmEx(
            cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
            QKV_D * HEADS, BATCHES, LATENT_D,
            &ONE,
            dQKVWeights + QKVWeightLayerOff, CUDA_R_32F, QKV_D * HEADS * LAYERS,
            dNorm, CUDA_R_32F, CONCAT_NORM_D,
            &ZERO,
            dQKV + QKVSeqOffset + QKVLayerOff, CUDA_R_32F, QKV_D * HEADS,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT
        );

        cublasGemmStridedBatchedEx(
            cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
            sequence + 1, 1, QUERY_D,
            &ROOT_QUERY_D,
            dQKV + QKVLayerOff + QUERY_D, CUDA_R_32F, QKV_D * HEADS * BATCHES, QKV_D,
            dQKV + QKVSeqOffset + QKVLayerOff, CUDA_R_32F, QKV_D * HEADS * BATCHES, QKV_D,
            &ZERO,
            dScore + scoreSeqOff, CUDA_R_32F, SEQUENCES, SEQUENCES * SEQUENCES,
            HEADS * BATCHES, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT
        );

        softmax(
            sequence + 1, HEADS * BATCHES,
            dScore + scoreSeqOff, SEQUENCES * SEQUENCES,
            dScore + scoreSeqOff, SEQUENCES * SEQUENCES
        );

        cublasGemmStridedBatchedEx(
            cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
            VALUE_D, 1, sequence + 1,
            &ONE,
            dQKV + QKVLayerOff + 2 * QUERY_D, CUDA_R_32F, QKV_D * HEADS * BATCHES, QKV_D,
            dScore + scoreSeqOff, CUDA_R_32F, SEQUENCES, SEQUENCES * SEQUENCES,
            &ZERO,
            dResult, CUDA_R_32F, VALUE_D * HEADS * BATCHES, VALUE_D,
            HEADS * BATCHES, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT
        );

        cudaMemcpy2D(
            dNorm + LATENT_D, CONCAT_NORM_D * sizeof(float),
            dResult, VALUE_D * HEADS * sizeof(float),
            VALUE_D * HEADS * sizeof(float), BATCHES, cudaMemcpyDeviceToDevice
        );

        cublasGemmEx(
            cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
            CONCAT_SWIGLU_D, BATCHES, CONCAT_NORM_D,
            &ONE,
            dSwigluWeights + swigluWeightLayerOff, CUDA_R_32F, CONCAT_SWIGLU_D,
            dNorm, CUDA_R_32F, CONCAT_NORM_D,
            &ZERO,
            dForward + forwardSeqOff + nextForwardLayerOff, CUDA_R_32F, CONCAT_SWIGLU_D,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT
        );

        residualSwiglu(
            LATENT_D, BATCHES,
            dForward + forwardSeqOff + nextForwardLayerOff,                CONCAT_SWIGLU_D, CONCAT_SWIGLU_D * BATCHES,
            dForward + forwardSeqOff + nextForwardLayerOff + LATENT_D,     CONCAT_SWIGLU_D, CONCAT_SWIGLU_D * BATCHES,
            dForward + forwardSeqOff + forwardLayerOff,                    CONCAT_SWIGLU_D, CONCAT_SWIGLU_D * BATCHES,
            1
        );
    }
}

void Model::sampleActions(int sequence) {
    const int forwardSeqOff = forwardSeqOffset(sequence);
    const int lastOff       = lastForwardLayerOffset();
    const int probOff       = probSeqOffset(sequence);
    const int actionOff     = BATCHES * sequence;

    softmaxSample(
        ACTIONS, BATCHES,
        dForward + forwardSeqOff + lastOff, CONCAT_SWIGLU_D,
        dProbs + probOff, ACTIONS,
        dActions + actionOff,
        *pSeed
    );
}

void Model::postSequences(const Environment& env, int epoch) {
    cudaMemcpy(
        dReward, env.reward(),
        BATCHES * SEQUENCES * sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemset(dBackwardTop, 0, CONCAT_SWIGLU_D * BATCHES * SEQUENCES * sizeof(float));

    gaeGradient(
        BATCHES, SEQUENCES, ACTIONS,
        dForward + CONCAT_SWIGLU_D * BATCHES * SEQUENCES * LAYERS + ACTIONS, CONCAT_SWIGLU_D,
        dProbs,
        dReward,
        dActions,
        dBackwardTop, CONCAT_SWIGLU_D,
        DISCOUNT_GAMMA, GAE_LAMBDA, ENTROPY_BETA
    );

    if ((epoch + 1) % LOG_SKIPS == 0) {
        for (int i = 0; i < BATCHES * SEQUENCES; i++) {
            averageReward += env.reward()[i];
        }
    }
}

void Model::backward() {
    for (int layer = LAYERS; layer--;) {
        const int nextForwardLayerOff  = CONCAT_SWIGLU_D * BATCHES * SEQUENCES * (layer + 1);
        const int forwardLayerOff      = CONCAT_SWIGLU_D * BATCHES * SEQUENCES * layer;
        const int QKVWeightLayerOff    = QKV_D * HEADS * layer;
        const int QKVLayerOff          = QKV_D * HEADS * BATCHES * SEQUENCES * layer;
        const int swigluWeightLayerOff = CONCAT_SWIGLU_D * CONCAT_NORM_D * layer;

        cublasGemmStridedBatchedEx(
            cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
            SEQUENCES, SEQUENCES, QUERY_D,
            &ROOT_QUERY_D,
            dQKV + QKVLayerOff + QUERY_D, CUDA_R_32F, QKV_D * HEADS * BATCHES, QKV_D,
            dQKV + QKVLayerOff,           CUDA_R_32F, QKV_D * HEADS * BATCHES, QKV_D,
            &ZERO,
            dScore, CUDA_R_32F, SEQUENCES, SEQUENCES * SEQUENCES,
            HEADS * BATCHES, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT
        );

        scoreMask(
            HEADS * BATCHES, SEQUENCES,
            dScore, SEQUENCES, SEQUENCES * SEQUENCES,
            dScore, SEQUENCES, SEQUENCES * SEQUENCES
        );

        softmax(
            SEQUENCES, SEQUENCES * HEADS * BATCHES,
            dScore, SEQUENCES,
            dScore, SEQUENCES
        );

        cublasGemmStridedBatchedEx(
            cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
            VALUE_D, SEQUENCES, SEQUENCES,
            &ONE,
            dQKV + QKVLayerOff + 2 * QUERY_D, CUDA_R_32F, QKV_D * HEADS * BATCHES, QKV_D,
            dScore, CUDA_R_32F, SEQUENCES, SEQUENCES * SEQUENCES,
            &ZERO,
            dResult, CUDA_R_32F, VALUE_D * HEADS * BATCHES, VALUE_D,
            HEADS * BATCHES, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT
        );

        residualSwigluGrad(
            LATENT_D, BATCHES * SEQUENCES,
            dBackwardTop,                 CONCAT_SWIGLU_D, CONCAT_SWIGLU_D * BATCHES,
            dBackwardTop + LATENT_D,      CONCAT_SWIGLU_D, CONCAT_SWIGLU_D * BATCHES,
            dBackwardBottom,              CONCAT_SWIGLU_D, CONCAT_SWIGLU_D * BATCHES,
            dForward + nextForwardLayerOff,                CONCAT_SWIGLU_D, CONCAT_SWIGLU_D * BATCHES,
            dForward + nextForwardLayerOff + LATENT_D,     CONCAT_SWIGLU_D, CONCAT_SWIGLU_D * BATCHES,
            dForward + forwardLayerOff,                    CONCAT_SWIGLU_D, CONCAT_SWIGLU_D * BATCHES,
            1
        );

        cublasGemmEx(
            cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
            CONCAT_NORM_D, BATCHES * SEQUENCES, CONCAT_SWIGLU_D,
            &ONE,
            dSwigluWeights + swigluWeightLayerOff, CUDA_R_32F, CONCAT_SWIGLU_D,
            dBackwardTop, CUDA_R_32F, CONCAT_SWIGLU_D,
            &ZERO,
            dNorm, CUDA_R_32F, CONCAT_NORM_D,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT
        );

        swap2D(
            VALUE_D * HEADS, BATCHES * SEQUENCES,
            dNorm + LATENT_D, CONCAT_NORM_D,
            dResult, VALUE_D * HEADS
        );

        cublasGemmStridedBatchedEx(
            cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
            VALUE_D, SEQUENCES, SEQUENCES,
            &ONE,
            dResult, CUDA_R_32F, VALUE_D * HEADS * BATCHES, VALUE_D,
            dScore,  CUDA_R_32F, SEQUENCES, SEQUENCES * SEQUENCES,
            &ZERO,
            dQKVGrad + 2 * QUERY_D, CUDA_R_32F, QKV_D * HEADS * BATCHES, QKV_D,
            HEADS * BATCHES, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT
        );

        cublasGemmStridedBatchedEx(
            cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
            SEQUENCES, SEQUENCES, VALUE_D,
            &ROOT_QUERY_D,
            dQKV + QKVLayerOff + 2 * QUERY_D, CUDA_R_32F, QKV_D * HEADS * BATCHES, QKV_D,
            dResult, CUDA_R_32F, VALUE_D * HEADS * BATCHES, VALUE_D,
            &ZERO,
            dScoreGrad, CUDA_R_32F, SEQUENCES, SEQUENCES * SEQUENCES,
            HEADS * BATCHES, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT
        );

        softmaxGrad(
            SEQUENCES, SEQUENCES * HEADS * BATCHES,
            dScore, SEQUENCES,
            dScoreGrad, SEQUENCES,
            dScoreGrad, SEQUENCES
        );

        cublasGemmStridedBatchedEx(
            cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
            QUERY_D, SEQUENCES, SEQUENCES,
            &ONE,
            dQKV + QKVLayerOff, CUDA_R_32F, QKV_D * HEADS * BATCHES, QKV_D,
            dScoreGrad, CUDA_R_32F, SEQUENCES, SEQUENCES * SEQUENCES,
            &ZERO,
            dQKVGrad + QUERY_D, CUDA_R_32F, QKV_D * HEADS * BATCHES, QKV_D,
            HEADS * BATCHES, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT
        );

        cublasGemmStridedBatchedEx(
            cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
            QUERY_D, SEQUENCES, SEQUENCES,
            &ONE,
            dQKV + QKVLayerOff + QUERY_D, CUDA_R_32F, QKV_D * HEADS * BATCHES, QKV_D,
            dScoreGrad, CUDA_R_32F, SEQUENCES, SEQUENCES * SEQUENCES,
            &ZERO,
            dQKVGrad, CUDA_R_32F, QKV_D * HEADS * BATCHES, QKV_D,
            HEADS * BATCHES, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT
        );

        cublasGemmEx(
            cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
            LATENT_D, BATCHES * SEQUENCES, QKV_D * HEADS,
            &ONE,
            dQKVWeights + QKVWeightLayerOff, CUDA_R_32F, QKV_D * HEADS * LAYERS,
            dQKVGrad, CUDA_R_32F, QKV_D * HEADS,
            &ONE,
            dNorm, CUDA_R_32F, CONCAT_NORM_D,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT
        );

        batchNormGrad(
            LATENT_D, BATCHES,
            dNorm, CONCAT_NORM_D, CONCAT_NORM_D * BATCHES,
            dForward + forwardLayerOff, CONCAT_SWIGLU_D, CONCAT_SWIGLU_D * BATCHES,
            dBackwardBottom, CONCAT_SWIGLU_D, CONCAT_SWIGLU_D * BATCHES,
            SEQUENCES
        );

        cublasGemmEx(
            cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
            QKV_D * HEADS, LATENT_D, BATCHES * SEQUENCES,
            &LR_NORM,
            dQKVGrad, CUDA_R_32F, QKV_D * HEADS,
            dNorm,    CUDA_R_32F, CONCAT_NORM_D,
            &ONE,
            dQKVWeightGrads + QKVWeightLayerOff, CUDA_R_32F, QKV_D * HEADS * LAYERS,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT
        );

        cublasGemmEx(
            cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
            CONCAT_SWIGLU_D, CONCAT_NORM_D, BATCHES * SEQUENCES,
            &LR_NORM,
            dBackwardTop, CUDA_R_32F, CONCAT_SWIGLU_D,
            dNorm,        CUDA_R_32F, CONCAT_NORM_D,
            &ONE,
            dSwigluWeightGrads + swigluWeightLayerOff, CUDA_R_32F, CONCAT_SWIGLU_D,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT
        );

        // swap
        float* tmp = dBackwardTop;
        dBackwardTop = dBackwardBottom;
        dBackwardBottom = tmp;
    }
}

void Model::epochLog(int epoch) const {
    if ((epoch + 1) % LOG_SKIPS == 0) {
        printf("Epoch %d: %.4f / %.4f\n",
            epoch + 1,
            averageReward / (BATCHES * BATCH_ITERATIONS * REWARD),
            1.5f * SEQUENCES / BOARD_D
        );
    }
}

void Model::update() {
    adamUpdate(
        QKV_D * HEADS * LAYERS, LATENT_D,
        dQKVWeightGrads,     QKV_D * HEADS * LAYERS,
        dQKVWeightGradMeans, QKV_D * HEADS * LAYERS,
        dQKVWeightGradVars,  QKV_D * HEADS * LAYERS,
        dQKVWeights,         QKV_D * HEADS * LAYERS,
        LR, MEAN_BETA, VAR_BETA, EPSILON
    );

    adamUpdate(
        CONCAT_SWIGLU_D, CONCAT_NORM_D * LAYERS,
        dSwigluWeightGrads,     CONCAT_SWIGLU_D,
        dSwigluWeightGradMeans, CONCAT_SWIGLU_D,
        dSwigluWeightGradVars,  CONCAT_SWIGLU_D,
        dSwigluWeights,         CONCAT_SWIGLU_D,
        LR, MEAN_BETA, VAR_BETA, EPSILON
    );
}

void Model::save(bool save_to_files) const {
    if (save_to_files) {
        saveDeviceTensor("qkv_weights.bin", dQKVWeights, QKV_D * HEADS * LAYERS * LATENT_D);
        saveDeviceTensor("qkv_weight_grad_means.bin", dQKVWeightGradMeans, QKV_D * HEADS * LAYERS * LATENT_D);
        saveDeviceTensor("qkv_weight_grad_vars.bin",  dQKVWeightGradVars,  QKV_D * HEADS * LAYERS * LATENT_D);
        saveDeviceTensor("swiglu_weights.bin", dSwigluWeights, CONCAT_SWIGLU_D * CONCAT_NORM_D * LAYERS);
        saveDeviceTensor("swiglu_weight_grad_means.bin", dSwigluWeightGradMeans, CONCAT_SWIGLU_D * CONCAT_NORM_D * LAYERS);
        saveDeviceTensor("swiglu_weight_grad_vars.bin",  dSwigluWeightGradVars,  CONCAT_SWIGLU_D * CONCAT_NORM_D * LAYERS);
    }
}

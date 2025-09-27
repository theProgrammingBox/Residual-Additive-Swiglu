#pragma once
#include <assert.h>
#include <stdio.h>
#include <ctime>
#include <cstring>        // for memset
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ======= Original configuration macros (unchanged) =======
#define EPOCHES (16 * 16 * 1 * 1)
#define LOG_SKIPS (1 * 16 * 1 * 1)
#define BIAS_BITS 1
#define TEMPORAL_BITS 10

#define LAYERS 8
#define SEQUENCES 32
#define BATCH_ITERATIONS 1
#define BATCHES 1024

#define ACTIONS 4
#define BOARD_D 8
#define BOARD_AREA (BOARD_D * BOARD_D)
#define OUTPUT_D (ACTIONS + 1)
#define LATENT_D 256
#define QUERY_D 64
#define VALUE_D 32
#define HEADS 4

#define CONCAT_SWIGLU_D (2 * LATENT_D)
#define CONCAT_NORM_D (LATENT_D + VALUE_D * HEADS)
#define QKV_D (2 * QUERY_D + VALUE_D)

#define LR 0.0001f
#define MEAN_BETA 0.9f
#define VAR_BETA 0.999f
#define EPSILON 1e-8f
#define DISCOUNT_GAMMA 0.99f
#define GAE_LAMBDA 0.95f
#define REWARD 1.0f
#define ENTROPY_BETA 0.01f

#define RENDER 0
#define LOAD 0
#define SAVE 1

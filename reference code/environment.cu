#include "environment.hpp"
#include "header.cuh"       // <-- ADD THIS LINE

// FORWARD DECLARATIONS (REMOVE THESE)
// int   hashI32(int& seed);
// float hashF32(int& seed, float lo, float hi);

Environment::Environment(int& seed) : pSeed(&seed) {
    // Host allocations (same sizes as original)
    hActions  = (int*)  malloc(BATCHES * sizeof(int));
    hPlayerX  = (int*)  malloc(BATCHES * sizeof(int));
    hPlayerY  = (int*)  malloc(BATCHES * sizeof(int));
    hGoalX    = (int*)  malloc(BATCHES * sizeof(int));
    hGoalY    = (int*)  malloc(BATCHES * sizeof(int));
    hStartIdx = (int*)  malloc(BATCHES * sizeof(int));
    hEndIdx   = (int*)  malloc(BATCHES * sizeof(int));
    hIdxQueue = (int*)  malloc(BOARD_AREA * BATCHES * sizeof(int));
    hBoard    = (float*)malloc(BOARD_AREA * BATCHES * sizeof(float));
    hReward   = (float*)malloc(BATCHES * SEQUENCES * sizeof(float));
}

Environment::~Environment() {
    free(hActions);
    free(hPlayerX);
    free(hPlayerY);
    free(hGoalX);
    free(hGoalY);
    free(hStartIdx);
    free(hEndIdx);
    free(hIdxQueue);
    free(hBoard);
    free(hReward);
}

void Environment::reset() {
    memset(hBoard,  0, BOARD_AREA * BATCHES * sizeof(float));
    memset(hReward, 0, BATCHES * SEQUENCES * sizeof(float));
    memset(hStartIdx, 0, BATCHES * sizeof(int));
    memset(hEndIdx,   0, BATCHES * sizeof(int));

    for (int batch = 0; batch < BATCHES; batch++) {
        hPlayerX[batch] = 0; // hashI32(*pSeed) % BOARD_D;
        hPlayerY[batch] = 0; // hashI32(*pSeed) % BOARD_D;
        hIdxQueue[BOARD_AREA * batch] = hPlayerX[batch] + BOARD_D * hPlayerY[batch];
        do {
            hGoalX[batch] = hashI32(*pSeed) % BOARD_D;
            hGoalY[batch] = hashI32(*pSeed) % BOARD_D;
        } while (hGoalX[batch] == hPlayerX[batch] && hGoalY[batch] == hPlayerY[batch]);
        hBoard[hPlayerX[batch] + BOARD_D * hPlayerY[batch] + BOARD_AREA * batch] = 1.0f;
        hBoard[hGoalX[batch]   + BOARD_D * hGoalY[batch]   + BOARD_AREA * batch] = -1.0f;
    }
}

void Environment::renderBoardIfNeeded(int epoch) const {
#if RENDER
    if (epoch == 0) {
        for (int y = 0; y < BOARD_D; y++) {
            for (int x = 0; x < BOARD_D; x++) {
                switch ((int)hBoard[x + BOARD_D * y]) {
                    case  0: printf("__"); break;
                    case  1: printf("[]"); break;
                    case -1: printf("@@"); break;
                }
            }
            printf("\n");
        }
    }
#else
    (void)epoch;
#endif
}

void Environment::step(const int* dActionsDevice, int sequence) {
    // Pull actions to host
    cudaMemcpy(hActions, dActionsDevice, BATCHES * sizeof(int), cudaMemcpyDeviceToHost);

    for (int batch = 0; batch < BATCHES; batch++) {
        int action = hActions[batch];
        int tmpStartIdx = hStartIdx[batch] + 1;
        tmpStartIdx *= tmpStartIdx != BOARD_AREA;
        if (tmpStartIdx != hEndIdx[batch]) {
            int newX = hPlayerX[batch];
            int newY = hPlayerY[batch];
            int oldX = newX;
            int oldY = newY;

            newX += (action == 3) * (newX < BOARD_D - 1) - (action == 1) * (newX > 0);
            newY += (action == 2) * (newY < BOARD_D - 1) - (action == 0) * (newY > 0);

            if (hBoard[newX + BOARD_D * newY + BOARD_AREA * batch] == 1.0f) {
                newX = oldX;
                newY = oldY;
            }
            int move = newX != hPlayerX[batch] || newY != hPlayerY[batch];
            int eat  = newX == hGoalX[batch]   && newY == hGoalY[batch];

            hStartIdx[batch] += move;
            hStartIdx[batch] *= hStartIdx[batch] != BOARD_AREA;
            hIdxQueue[hStartIdx[batch] + BOARD_AREA * batch] = newX + BOARD_D * newY;

            hBoard[newX + BOARD_D * newY + BOARD_AREA * batch] = 1.0f;
            hBoard[hIdxQueue[hEndIdx[batch] + BOARD_AREA * batch] + BOARD_AREA * batch] = !move || eat;

            hEndIdx[batch] += move && !eat;
            hEndIdx[batch] *= hEndIdx[batch] != BOARD_AREA;

            hPlayerX[batch] = newX;
            hPlayerY[batch] = newY;

            if (eat) {
                hReward[batch + BATCHES * sequence] = REWARD;
                tmpStartIdx = hStartIdx[batch] + 1;
                tmpStartIdx *= tmpStartIdx != BOARD_AREA;
                if (tmpStartIdx != hEndIdx[batch]) {
                    do {
                        hGoalX[batch] = hashI32(*pSeed) % BOARD_D;
                        hGoalY[batch] = hashI32(*pSeed) % BOARD_D;
                    } while (hBoard[hGoalX[batch] + BOARD_D * hGoalY[batch] + BOARD_AREA * batch] != 0.0f);
                    hBoard[hGoalX[batch] + BOARD_D * hGoalY[batch] + BOARD_AREA * batch] = -1.0f;
                }
            }
        }
    }
}

void Environment::renderStepIfNeeded(const float* dProbsDevice,
                                     const float* dForwardDevice,
                                     int epoch, int sequence,
                                     int forwardSeqOffset,
                                     int lastForwardLayerOffset,
                                     int probSeqOffset) const {
#if RENDER
    if (epoch == 0) {
        float probs[ACTIONS];
        float value;
        cudaMemcpy(probs, dProbsDevice + probSeqOffset, ACTIONS * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&value, dForwardDevice + forwardSeqOffset + lastForwardLayerOffset + ACTIONS,
                   sizeof(float), cudaMemcpyDeviceToHost);
        printf("Epoch %d, Sequence %d: Actions Probs:", epoch + 1, sequence + 1);
        for (int i = 0; i < ACTIONS; i++) printf("%.4f ", probs[i]);
        printf(", Value: %.4f\n", value);
        getchar();
        for (int y = 0; y < BOARD_D; y++) {
            for (int x = 0; x < BOARD_D; x++) {
                switch ((int)hBoard[x + BOARD_D * y]) {
                    case  0: printf("__"); break;
                    case  1: printf("[]"); break;
                    case -1: printf("@@"); break;
                }
            }
            printf("\n");
        }
    }
#else
    (void)dProbsDevice; (void)dForwardDevice; (void)epoch; (void)sequence;
    (void)forwardSeqOffset; (void)lastForwardLayerOffset; (void)probSeqOffset;
#endif
}
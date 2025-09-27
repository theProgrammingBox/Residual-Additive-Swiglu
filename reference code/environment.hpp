#pragma once
#include "config.cuh"

// A simple, batched gridworld environment with the same state & transitions
// as the original code. Host-side state is kept identical.
class Environment {
public:
    explicit Environment(int& seed);
    ~Environment();

    // Resets host buffers for a new batch iteration and initializes positions/goals.
    void reset();

    // Optional first-frame render (same output & condition as original).
    void renderBoardIfNeeded(int epoch) const;

    // Applies sampled actions for the given sequence, updates board & rewards.
    // dActionsDevice must point to device array of length BATCHES for this sequence.
    void step(const int* dActionsDevice, int sequence);

    // Optional per-step render (reads from model device buffers to print).
    void renderStepIfNeeded(const float* dProbsDevice,
                            const float* dForwardDevice,
                            int epoch,
                            int sequence,
                            int forwardSeqOffset,
                            int lastForwardLayerOffset,
                            int probSeqOffset) const;

    // Host buffers exposed for the model I/O (unchanged data layout).
    float* board()  const { return hBoard;  }
    float* reward() const { return hReward; }

private:
    // Host state (identical to original)
    int*   hActions;
    int*   hPlayerX;
    int*   hPlayerY;
    int*   hGoalX;
    int*   hGoalY;
    int*   hStartIdx;
    int*   hEndIdx;
    int*   hIdxQueue;
    float* hBoard;
    float* hReward;

    int* pSeed; // shared RNG seed (matches original hashI32(seed) usage)
};

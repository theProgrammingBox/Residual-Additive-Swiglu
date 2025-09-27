#pragma once
#include "config.cuh"
#include "environment.hpp"

// A lightweight "model" abstraction reminiscent of Python AI libs:
// - forwardSequence(): per-timestep forward pass (includes attention + MLP)
// - sampleActions():    samples actions (policy head) on device
// - postSequences():    uploads rewards & computes advantages
// - backward():         full backprop for the block
// - update():           Adam updates
class Model {
public:
    Model(int& seed, cublasHandle_t handle);
    ~Model();

    void initWeights(bool load_from_files);
    void positionEmbedding();  // calls the global function via ::positionEmbedding

    void zeroGradsAndResetLogging();

    // Forward for one sequence step: copies env.board() to device and runs the layer stack.
    void forwardSequence(int sequence, const Environment& env);

    // Samples actions for a given sequence (writes to dActions).
    void sampleActions(int sequence);

    // After finishing all sequences in an iteration: uploads rewards and computes policy/value grads.
    void postSequences(const Environment& env, int epoch);

    // Backprop through all layers (full block), accumulates weight grads.
    void backward();

    // Prints the same scalar as original logging.
    void epochLog(int epoch) const;

    // Adam updates (weights only).
    void update();

    // Save (unchanged).
    void save(bool save_to_files) const;

    // Accessors for env rendering/stepping.
    const int*   d_actions()  const { return dActions; }
    const float* d_probs()    const { return dProbs; }
    const float* d_forward()  const { return dForward; }

    // Helpers for offsets used by rendering.
    static int forwardSeqOffset(int sequence) { return CONCAT_SWIGLU_D * BATCHES * sequence; }
    static int probSeqOffset(int sequence)    { return ACTIONS * BATCHES * sequence; }
    static int lastForwardLayerOffset()       { return CONCAT_SWIGLU_D * BATCHES * SEQUENCES * LAYERS; }

private:
    // Device tensors (identical shapes to original)
    int*   dActions;
    float* dProbs;
    float* dReward;

    float* dForward;
    float* dNorm;
    float* dResult;
    float* dQKV;
    float* dScore;

    float* dBackwardTop;
    float* dBackwardBottom;
    float* dQKVGrad;
    float* dScoreGrad;

    float* dQKVWeights;
    float* dQKVWeightGrads;
    float* dQKVWeightGradMeans;
    float* dQKVWeightGradVars;

    float* dSwigluWeights;
    float* dSwigluWeightGrads;
    float* dSwigluWeightGradMeans;
    float* dSwigluWeightGradVars;

    // State
    float averageReward;
    int*   pSeed;
    cublasHandle_t cublasHandle;

    // Constants used by cublas calls (match original)
    const float ONE;
    const float ZERO;
    const float ROOT_QUERY_D;
    const float LR_NORM;
};

#include "config.cuh"
#include "environment.hpp"
#include "model.hpp"

int main() {
    // Same asserts as before
    assert(BOARD_AREA + BIAS_BITS + TEMPORAL_BITS <= LATENT_D);
    assert(OUTPUT_D <= LATENT_D);

    // Shared RNG and cuBLAS
    int seed = 0;
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    // Modules
    Environment env(seed);
    Model       model(seed, cublasHandle);

    // Init model (weights) and positional embeddings (unchanged)
    model.initWeights(LOAD != 0);
    model.positionEmbedding();

    // Timing (unchanged)
    time_t start_time, end_time;
    time(&start_time);

    for (int epoch = 0; epoch < EPOCHES; epoch++) {
        model.zeroGradsAndResetLogging();

        for (int batchIteration = 0; batchIteration < BATCH_ITERATIONS; batchIteration++) {
            env.reset();
            env.renderBoardIfNeeded(epoch);

            for (int sequence = 0; sequence < SEQUENCES; sequence++) {
                // Forward & action sampling (unchanged ordering)
                model.forwardSequence(sequence, env);
                model.sampleActions(sequence);

                // Apply actions to environment; may render step
                const int fwdOff  = Model::forwardSeqOffset(sequence);
                const int lastOff = Model::lastForwardLayerOffset();
                const int probOff = Model::probSeqOffset(sequence);

                env.step(model.d_actions() + BATCHES * sequence, sequence);
                env.renderStepIfNeeded(model.d_probs(), model.d_forward(),
                                       epoch, sequence, fwdOff, lastOff, probOff);
            }

            // Post sequences: upload rewards & compute policy/value grads, then backward
            model.postSequences(env, epoch);
            model.backward();
        }

        // Log and update (unchanged)
        model.epochLog(epoch);
        model.update();
    }

    time(&end_time);
    printf("Simulation time: %ld seconds\n", end_time - start_time);

    model.save(SAVE != 0);

    cublasDestroy(cublasHandle);
    return 0;
}

#include "utils.cuh"
#include "src/GridworldEnv.h"

// ... #defines ...

int main() {
    int seed = 0;
    // ... other setup ...

    // Create an instance of our environment
    GridworldEnv env(BATCHES, BOARD_D, seed);

    // Allocate DEVICE buffers here (all the d* buffers)
    int* dActions = mallocDeviceTensor<int>(BATCHES * SEQUENCES);
    // ... etc ...

    for (int epoch = 0; epoch < EPOCHES; epoch++) {
        for (int batchIteration = 0; batchIteration < BATCH_ITERATIONS; batchIteration++) {
            env.reset();

            // You'll need a cudaMemcpy here to get the state to the GPU
            // cudaMemcpy(dForward, env.get_state(), ...);

            for (int sequence = 0; sequence < SEQUENCES; sequence++) {
                // ... AI FORWARD PASS LOGIC ...

                // Copy actions from GPU to some host buffer `hActions`
                // cudaMemcpy(hActions, dActions + offset, ...);

                // Tell the environment to take a step
                env.step(hActions);

                // Copy new state and rewards to GPU
                // cudaMemcpy(dForward + offset, env.get_state(), ...);
                // cudaMemcpy(dReward, env.get_rewards(), ...);
            }
            // ... AI BACKWARD PASS & UPDATE LOGIC ...
        }
    }
    return 0;
}
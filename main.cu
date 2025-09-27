#include "config.hpp"
#include "environment.hpp"

#include <cstdio>
#include <vector>

int main() {
    TrainingConfig config;
    config.input_bits = 4;
    config.output_bits = 6;
    config.batch_size = 4;
    config.task = TrainingTask::Add;

    Environment env(config);

    std::vector<float> inputs;
    std::vector<float> targets;
    env.generateBatch(inputs, targets);

    std::printf("Generated %zu samples (input width %zu, target width %zu)\n",
                config.batch_size, env.inputWidth(), env.outputWidth());

    return 0;
}

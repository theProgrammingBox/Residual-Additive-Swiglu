#pragma once

#include <cstddef>
#include <cstdint>

// Training configuration shared across host components.

enum class TrainingTask {
    Add,
    Concat,
    Multiply,
};

struct TrainingConfig {
    std::size_t input_bits = 4;        // bits per input number
    std::size_t output_bits = 5;       // bits in the target representation
    std::size_t hidden_dim = 32;       // residual width of the model
    std::size_t swiglu_components = 1; // multi-component SwiGLU factor (1 == legacy)
    std::size_t batch_size = 32;
    std::uint64_t seed = 0x1234abcdULL;
    TrainingTask task = TrainingTask::Add;
};

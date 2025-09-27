#include "config.hpp"
#include "environment.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>

namespace {
int bitAt(std::uint64_t value, std::size_t bit) {
    return static_cast<int>((value >> bit) & 1ULL);
}

void testAddition() {
    TrainingConfig cfg;
    cfg.input_bits = 3;
    cfg.output_bits = 4;
    cfg.task = TrainingTask::Add;
    cfg.batch_size = 8;

    Environment env(cfg);
    const std::size_t total = 1ULL << (cfg.input_bits * 2);
    for (std::size_t idx = 0; idx < total; ++idx) {
        auto sample = env.generateSample(idx);
        const std::uint64_t expected = sample.lhs + sample.rhs;
        for (std::size_t bit = 0; bit < cfg.output_bits; ++bit) {
            assert(sample.target_bits[bit] == bitAt(expected, bit));
        }
    }
}

void testOverflowClamp() {
    TrainingConfig cfg;
    cfg.input_bits = 4;
    cfg.output_bits = 3;
    cfg.task = TrainingTask::Add;

    Environment env(cfg);
    auto sample = env.generateSample((15ULL << cfg.input_bits) | 15ULL);
    // 15 + 15 = 30 -> binary 11110, expect lower 3 bits (110)
    assert(sample.target_bits.size() == cfg.output_bits);
    assert(sample.target_bits[0] == bitAt(30, 0));
    assert(sample.target_bits[1] == bitAt(30, 1));
    assert(sample.target_bits[2] == bitAt(30, 2));
}

void testConcat() {
    TrainingConfig cfg;
    cfg.input_bits = 2;
    cfg.output_bits = 4;
    cfg.task = TrainingTask::Concat;

    Environment env(cfg);
    auto sample = env.generateSample(0b1111);
    std::uint64_t concatenated = (sample.rhs << cfg.input_bits) | sample.lhs;
    for (std::size_t bit = 0; bit < cfg.output_bits; ++bit) {
        assert(sample.target_bits[bit] == bitAt(concatenated, bit));
    }
}

void testMultiply() {
    TrainingConfig cfg;
    cfg.input_bits = 3;
    cfg.output_bits = 6;
    cfg.task = TrainingTask::Multiply;

    Environment env(cfg);
    const std::size_t total = 1ULL << (cfg.input_bits * 2);
    for (std::size_t idx = 0; idx < total; ++idx) {
        auto sample = env.generateSample(idx);
        const std::uint64_t expected = sample.lhs * sample.rhs;
        for (std::size_t bit = 0; bit < cfg.output_bits; ++bit) {
            assert(sample.target_bits[bit] == bitAt(expected, bit));
        }
    }
}
}

int main() {
    testAddition();
    testOverflowClamp();
    testConcat();
    testMultiply();

    std::cout << "Environment tests passed.\n";
    return 0;
}

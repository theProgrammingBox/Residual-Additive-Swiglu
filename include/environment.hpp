#pragma once

#include "config.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

struct EnvironmentSample {
    std::vector<int> input_bits;
    std::vector<int> target_bits;
    std::uint64_t lhs = 0;
    std::uint64_t rhs = 0;
};

class Environment {
public:
    explicit Environment(TrainingConfig config);

    const TrainingConfig& config() const noexcept { return m_config; }

    void generateBatch(std::vector<float>& inputs, std::vector<float>& targets);

    EnvironmentSample generateSample(std::uint64_t index) const;

    std::size_t inputWidth() const noexcept { return 2 * m_config.input_bits; }
    std::size_t outputWidth() const noexcept { return m_config.output_bits; }

private:
    TrainingConfig m_config;
    std::mt19937_64 m_rng;
    std::uniform_int_distribution<std::uint64_t> m_valueDist;

    static std::uint64_t maskForBits(std::size_t bits);
    static void packBits(std::uint64_t value, std::size_t bits, std::vector<float>& dst, std::size_t offset);

    std::pair<std::uint64_t, std::uint64_t> sampleValues();
    std::uint64_t computeTargetValue(std::uint64_t lhs, std::uint64_t rhs) const;
    void fillInput(std::uint64_t lhs, std::uint64_t rhs, std::vector<float>& dst, std::size_t offset) const;
    void fillTarget(std::uint64_t lhs, std::uint64_t rhs, std::vector<float>& dst, std::size_t offset) const;
};

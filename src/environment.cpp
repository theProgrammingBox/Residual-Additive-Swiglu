#include "environment.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace {
constexpr std::size_t kInputCount = 2; // number of operands
constexpr std::size_t kMaxSupportedBits = 31; // allows results to fit into 64-bit
}

Environment::Environment(TrainingConfig config)
    : m_config(std::move(config)),
      m_rng(static_cast<std::mt19937_64::result_type>(m_config.seed)),
      m_valueDist(0, maskForBits(m_config.input_bits)) {
    if (m_config.input_bits == 0 || m_config.input_bits > kMaxSupportedBits) {
        throw std::invalid_argument("input_bits must be in [1, 31]");
    }
    if (m_config.output_bits == 0 || m_config.output_bits > 64) {
        throw std::invalid_argument("output_bits must be in [1, 64]");
    }
    if (m_config.swiglu_components == 0) {
        throw std::invalid_argument("swiglu_components must be > 0");
    }
    if (m_config.hidden_dim == 0) {
        throw std::invalid_argument("hidden_dim must be > 0");
    }
    if (m_config.batch_size == 0) {
        throw std::invalid_argument("batch_size must be > 0");
    }
}

void Environment::generateBatch(std::vector<float>& inputs, std::vector<float>& targets) {
    inputs.assign(m_config.batch_size * inputWidth(), 0.0f);
    targets.assign(m_config.batch_size * outputWidth(), 0.0f);

    for (std::size_t i = 0; i < m_config.batch_size; ++i) {
        auto [lhs, rhs] = sampleValues();
        const std::size_t inputOffset = i * inputWidth();
        const std::size_t targetOffset = i * outputWidth();
        fillInput(lhs, rhs, inputs, inputOffset);
        fillTarget(lhs, rhs, targets, targetOffset);
    }
}

EnvironmentSample Environment::generateSample(std::uint64_t index) const {
    const std::uint64_t mask = maskForBits(m_config.input_bits);
    const std::uint64_t lhs = index & mask;
    const std::uint64_t rhs = (index >> m_config.input_bits) & mask;

    EnvironmentSample sample;
    sample.lhs = lhs;
    sample.rhs = rhs;
    sample.input_bits.assign(inputWidth(), 0);
    sample.target_bits.assign(outputWidth(), 0);

    std::vector<float> tmpInput(inputWidth());
    std::vector<float> tmpTarget(outputWidth());
    fillInput(lhs, rhs, tmpInput, 0);
    fillTarget(lhs, rhs, tmpTarget, 0);

    for (std::size_t i = 0; i < inputWidth(); ++i) {
        sample.input_bits[i] = static_cast<int>(tmpInput[i]);
    }
    for (std::size_t i = 0; i < outputWidth(); ++i) {
        sample.target_bits[i] = static_cast<int>(tmpTarget[i]);
    }

    return sample;
}

std::uint64_t Environment::maskForBits(std::size_t bits) {
    if (bits >= 64) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return (std::uint64_t{1} << bits) - 1;
}

void Environment::packBits(std::uint64_t value, std::size_t bits, std::vector<float>& dst, std::size_t offset) {
    for (std::size_t i = 0; i < bits && (offset + i) < dst.size(); ++i) {
        dst[offset + i] = static_cast<float>((value >> i) & 1ULL);
    }
}

std::pair<std::uint64_t, std::uint64_t> Environment::sampleValues() {
    return {m_valueDist(m_rng), m_valueDist(m_rng)};
}

std::uint64_t Environment::computeTargetValue(std::uint64_t lhs, std::uint64_t rhs) const {
    switch (m_config.task) {
    case TrainingTask::Add:
        return lhs + rhs;
    case TrainingTask::Multiply:
        return lhs * rhs;
    case TrainingTask::Concat: {
        const std::uint64_t shifted = rhs << m_config.input_bits;
        return shifted | lhs;
    }
    default:
        return 0;
    }
}

void Environment::fillInput(std::uint64_t lhs, std::uint64_t rhs, std::vector<float>& dst, std::size_t offset) const {
    packBits(lhs, m_config.input_bits, dst, offset);
    packBits(rhs, m_config.input_bits, dst, offset + m_config.input_bits);
}

void Environment::fillTarget(std::uint64_t lhs, std::uint64_t rhs, std::vector<float>& dst, std::size_t offset) const {
    if (m_config.task == TrainingTask::Concat) {
        const std::uint64_t concatenated = (rhs << m_config.input_bits) | lhs;
        packBits(concatenated, m_config.output_bits, dst, offset);
    } else {
        const std::uint64_t value = computeTargetValue(lhs, rhs);
        packBits(value, m_config.output_bits, dst, offset);
    }
}

#include "GridworldEnv.h"
#include "../utils.cuh" // Need this for hashI32
#include <cstring>     // For memset

GridworldEnv::GridworldEnv(int batch_size, int board_dim, int& seed)
    : m_batch_size(batch_size), m_board_dim(board_dim), m_board_area(board_dim * board_dim), m_seed(seed) {
    // Use malloc to allocate all the h* buffers here, just like in main()
    hPlayerX = (int*)malloc(m_batch_size * sizeof(int));
    // ... malloc all other buffers ...
    hBoard = (float*)malloc(m_board_area * m_batch_size * sizeof(float));
    hReward = (float*)malloc(m_batch_size * 128 * sizeof(float)); // SEQUENCES
}

GridworldEnv::~GridworldEnv() {
    // Use free() to clean up all the h* buffers
    free(hPlayerX);
    // ... free all other buffers ...
}

void GridworldEnv::reset() {
    // Move the environment reset logic from main's batchIteration loop here
    memset(hBoard, 0, m_board_area * m_batch_size * sizeof(float));
    // ... etc ...
    for (int batch = 0; batch < m_batch_size; batch++) {
        // ... logic to place player and goal ...
    }
}

void GridworldEnv::step(const int* h_actions) {
    // Move the game step logic from main's sequence loop here
    // The logic that starts with "for (int batch = 0; batch < BATCHES; batch++)"
    // and updates hPlayerX, hBoard, hReward based on h_actions.
}
#pragma once // Prevents this header from being included multiple times
#include <vector>

class GridworldEnv {
public:
    // Constructor: sets up the environment
    GridworldEnv(int batch_size, int board_dim, int& seed);
    // Destructor: cleans up memory
    ~GridworldEnv();

    void reset();
    void step(const int* h_actions);

    // Getters to access the state from outside
    const float* get_state() const { return hBoard; }
    const float* get_rewards() const { return hReward; }

private:
    int m_batch_size;
    int m_board_dim;
    int m_board_area;
    int& m_seed; // Use a reference to the main seed

    // All host-side game buffers
    int* hPlayerX;
    int* hPlayerY;
    // ... all other h* buffers from main ...
    float* hBoard;
    float* hReward;
};
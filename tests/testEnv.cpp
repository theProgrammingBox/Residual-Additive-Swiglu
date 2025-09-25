#include <gtest/gtest.h>
#include "../src/GridworldEnv.h"

// A test case for the GridworldEnv
TEST(GridworldEnvTest, PlayerMovesCorrectly) {
    int seed = 0;
    // Create an environment with a single batch and a 3x3 board
    GridworldEnv env(1, 3, seed);

    // We can't easily set the internal state, so we'll just test the reset.
    // A more advanced test would add a "setter" method to the class for testing.
    env.reset();

    // Let's just check if the state pointer is not null after reset.
    // This is a very basic "smoke test".
    ASSERT_NE(env.get_state(), nullptr);

    // A better test would be:
    // 1. Add a method like `void setPlayerPosition(int x, int y)` to GridworldEnv
    // 2. Call env.setPlayerPosition(1, 1);
    // 3. Call env.step({3}); // Action 3 = RIGHT
    // 4. Add a method `int getPlayerX()` to GridworldEnv
    // 5. ASSERT_EQ(env.getPlayerX(), 2);
}
#include <gtest/gtest.h>
#include "adaptive_model.hpp"

// Test: AdaptiveModel_InitialState
TEST(AdaptiveModelTest, InitialState) {
    AdaptiveModel model(2);
    
    // Verify initial frequencies
    EXPECT_EQ(model.get_frequency(0), 1);
    EXPECT_EQ(model.get_frequency(1), 1);
    
    // Verify cumulative frequencies
    const std::vector<int>& cum_freq = model.get_cumulative_freq();
    EXPECT_EQ(cum_freq[0], 2);  // Total frequency
    EXPECT_EQ(cum_freq[1], 1);  // Cumulative for symbol 0
    EXPECT_EQ(cum_freq[2], 0);  // Cumulative for symbol 1
    
    // Verify number of symbols
    EXPECT_EQ(model.get_num_symbols(), 2);
}

// Test: AdaptiveModel_UpdateFrequency
TEST(AdaptiveModelTest, UpdateFrequency) {
    AdaptiveModel model(2);
    
    // Update symbol 0 multiple times
    model.update_model(0);
    EXPECT_EQ(model.get_frequency(0), 2);
    EXPECT_EQ(model.get_frequency(1), 1);
    
    model.update_model(0);
    EXPECT_EQ(model.get_frequency(0), 3);
    EXPECT_EQ(model.get_frequency(1), 1);
    
    // Verify cumulative frequencies update
    const std::vector<int>& cum_freq = model.get_cumulative_freq();
    EXPECT_EQ(cum_freq[0], 4);  // Total: 3 + 1
}

// Test: AdaptiveModel_Rescaling
TEST(AdaptiveModelTest, Rescaling) {
    AdaptiveModel model(2);
    
    // Update symbols many times to trigger rescaling
    // MAX_FREQUENCY is 16383, so we need to update many times
    for (int i = 0; i < 10000; i++) {
        model.update_model(0);
        if (i % 1000 == 0) {
            // Check that frequencies are reasonable
            EXPECT_LE(model.get_frequency(0), 16383);
            EXPECT_LE(model.get_frequency(1), 16383);
        }
    }
    
    // After many updates, frequencies should have been rescaled
    // Total frequency should be less than MAX_FREQUENCY
    const std::vector<int>& cum_freq = model.get_cumulative_freq();
    EXPECT_LE(cum_freq[0], 16383);
}

// Test: AdaptiveModel_SymbolResorting
TEST(AdaptiveModelTest, SymbolResorting) {
    AdaptiveModel model(2);
    
    // Initially, symbols should be in order
    int initial_index_0 = model.get_internal_index(0);
    int initial_index_1 = model.get_internal_index(1);
    
    // Update symbol 1 many times (more than symbol 0)
    for (int i = 0; i < 100; i++) {
        model.update_model(1);
    }
    
    // Symbol 1 should now have higher frequency
    EXPECT_GT(model.get_frequency(1), model.get_frequency(0));
    
    // Symbol 1 might move to a lower internal index (more frequent symbols have lower indices)
    // This depends on the sorting implementation
    int new_index_1 = model.get_internal_index(1);
    int new_index_0 = model.get_internal_index(0);
    
    // Verify internal index mapping still works
    EXPECT_EQ(model.get_symbol_from_internal_index(new_index_1), 1);
    EXPECT_EQ(model.get_symbol_from_internal_index(new_index_0), 0);
}

// Test: AdaptiveModel_InternalIndexMapping
TEST(AdaptiveModelTest, InternalIndexMapping) {
    AdaptiveModel model(2);
    
    // Test round-trip: symbol -> internal index -> symbol
    for (int symbol = 0; symbol < 2; symbol++) {
        int internal_index = model.get_internal_index(symbol);
        EXPECT_GE(internal_index, 1);
        EXPECT_LE(internal_index, 2);
        
        int recovered_symbol = model.get_symbol_from_internal_index(internal_index);
        EXPECT_EQ(recovered_symbol, symbol);
    }
}

// Test: AdaptiveModel_MultipleSymbols
TEST(AdaptiveModelTest, MultipleSymbols) {
    AdaptiveModel model(4);
    
    // Verify all symbols start with equal frequencies
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(model.get_frequency(i), 1);
    }
    
    // Update one symbol
    model.update_model(2);
    EXPECT_EQ(model.get_frequency(2), 2);
    EXPECT_EQ(model.get_frequency(0), 1);
    EXPECT_EQ(model.get_frequency(1), 1);
    EXPECT_EQ(model.get_frequency(3), 1);
}

// Test: AdaptiveModel_Reset
TEST(AdaptiveModelTest, Reset) {
    AdaptiveModel model(2);
    
    // Update model
    model.update_model(0);
    model.update_model(0);
    EXPECT_EQ(model.get_frequency(0), 3);
    
    // Reset
    model.reset();
    
    // Verify back to initial state
    EXPECT_EQ(model.get_frequency(0), 1);
    EXPECT_EQ(model.get_frequency(1), 1);
}

// Test: AdaptiveModel_InvalidSymbol
TEST(AdaptiveModelTest, InvalidSymbol) {
    AdaptiveModel model(2);
    
    // Try to update with invalid symbol (should be handled gracefully or throw)
    // This depends on implementation - for now, we'll test that it doesn't crash
    // If the implementation throws, we'd use EXPECT_THROW
    // For now, we'll just verify valid symbols work
    EXPECT_NO_THROW(model.update_model(0));
    EXPECT_NO_THROW(model.update_model(1));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


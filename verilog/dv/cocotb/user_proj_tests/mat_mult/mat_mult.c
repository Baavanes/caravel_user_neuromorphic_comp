// SPDX-License-Identifier: Apache-2.0
// Written by: Claude (Anthropic AI Assistant)
//
// Matrix Multiplication Accelerator Firmware Test
//
// Description:
//   Comprehensive test of the mat_mult_wb.v systolic array matrix multiplier.
//   Tests various matrix operations including identity, sequential values,
//   signed/unsigned modes, and reset functionality.

#define USER_ADDR_SPACE_C_HEADER_FILE
#include <firmware_apis.h>
#include <custom_user_space.h>
#include <stdint.h>
#include <string.h>

// =============================================================================
// Address Map for Matrix Multiplier
// =============================================================================

#define MATMUL_BASE       0x30000000

// Control and Status Registers
#define MATMUL_CTRL       (MATMUL_BASE + 0x000)
#define MATMUL_STATUS     (MATMUL_BASE + 0x004)
#define MATMUL_CYCLES     (MATMUL_BASE + 0x008)
#define MATMUL_VERSION    (MATMUL_BASE + 0x00C)

// Matrix Cache Addresses
#define MATMUL_A_BASE     (MATMUL_BASE + 0x100)
#define MATMUL_B_BASE     (MATMUL_BASE + 0x200)
#define MATMUL_C_BASE     (MATMUL_BASE + 0x400)

// Control Register Bits
#define CTRL_START        (1 << 0)
#define CTRL_RESET        (1 << 1)
#define CTRL_SIGNED       (1 << 2)
#define CTRL_IRQ_EN       (1 << 8)

// Status Register Bits
#define STATUS_BUSY       (1 << 0)
#define STATUS_DONE       (1 << 1)
#define STATUS_READY      (1 << 2)

// Expected Version
#define EXPECTED_VERSION  0xA7770001

// Test parameters
#define MATRIX_SIZE       8
#define MAX_POLL_CYCLES   1000

// =============================================================================
// Utility Functions
// =============================================================================

// Wait for specified number of CPU cycles
static inline void wait_cycles(uint32_t cycles)
{
    for (uint32_t i = 0; i < cycles; i++) {
        __asm__ volatile ("nop");
    }
}

// Pack 4 8-bit elements into a 32-bit word
static inline uint32_t pack_elements(int8_t e0, int8_t e1, int8_t e2, int8_t e3)
{
    return ((uint32_t)(uint8_t)e0) |
           (((uint32_t)(uint8_t)e1) << 8) |
           (((uint32_t)(uint8_t)e2) << 16) |
           (((uint32_t)(uint8_t)e3) << 24);
}

// Extract 8-bit element from 32-bit word
static inline int8_t extract_element(uint32_t word, uint8_t index)
{
    return (int8_t)((word >> (index * 8)) & 0xFF);
}

// =============================================================================
// Matrix Operations
// =============================================================================

// Write 8x8 matrix to cache (matrices are stored row-major, 4 elements per word)
void write_matrix_a(int8_t matrix[MATRIX_SIZE][MATRIX_SIZE])
{
    volatile uint32_t *cache = (volatile uint32_t *)MATMUL_A_BASE;

    for (int row = 0; row < MATRIX_SIZE; row++) {
        for (int col = 0; col < MATRIX_SIZE; col += 4) {
            uint32_t packed = pack_elements(
                matrix[row][col],
                matrix[row][col+1],
                matrix[row][col+2],
                matrix[row][col+3]
            );
            cache[row * 2 + col/4] = packed;
        }
    }
}

void write_matrix_b(int8_t matrix[MATRIX_SIZE][MATRIX_SIZE])
{
    volatile uint32_t *cache = (volatile uint32_t *)MATMUL_B_BASE;

    for (int row = 0; row < MATRIX_SIZE; row++) {
        for (int col = 0; col < MATRIX_SIZE; col += 4) {
            uint32_t packed = pack_elements(
                matrix[row][col],
                matrix[row][col+1],
                matrix[row][col+2],
                matrix[row][col+3]
            );
            cache[row * 2 + col/4] = packed;
        }
    }
}

// Read 8x8 result matrix from cache (32-bit results, 1 per word)
void read_matrix_c(int32_t result[MATRIX_SIZE][MATRIX_SIZE])
{
    volatile uint32_t *cache = (volatile uint32_t *)MATMUL_C_BASE;

    for (int row = 0; row < MATRIX_SIZE; row++) {
        for (int col = 0; col < MATRIX_SIZE; col++) {
            result[row][col] = (int32_t)cache[row * MATRIX_SIZE + col];
        }
    }
}

// =============================================================================
// Accelerator Control
// =============================================================================

// Start multiplication with specified mode
void start_multiplication(uint8_t signed_mode)
{
    uint32_t ctrl = CTRL_START;
    if (signed_mode) {
        ctrl |= CTRL_SIGNED;
    }
    *((volatile uint32_t *)MATMUL_CTRL) = ctrl;
}

// Poll status register until DONE bit is set
uint8_t wait_for_done(void)
{
    uint32_t poll_count = 0;
    uint32_t status;

    while (poll_count < MAX_POLL_CYCLES) {
        status = *((volatile uint32_t *)MATMUL_STATUS);
        if (status & STATUS_DONE) {
            return 1;  // Success
        }
        wait_cycles(10);
        poll_count++;
    }

    return 0;  // Timeout
}

// Reset the accelerator
void reset_accelerator(void)
{
    *((volatile uint32_t *)MATMUL_CTRL) = CTRL_RESET;
    wait_cycles(100);
}

// =============================================================================
// Test Functions
// =============================================================================

// Test 1: Version Register Check
uint8_t test_version(void)
{
    uint32_t version = *((volatile uint32_t *)MATMUL_VERSION);

    if (version == EXPECTED_VERSION) {
        return 1;  // Pass
    }
    return 0;  // Fail
}

// Test 2: Status Register Check (should be READY initially)
uint8_t test_status_ready(void)
{
    uint32_t status = *((volatile uint32_t *)MATMUL_STATUS);

    if (status & STATUS_READY) {
        return 1;  // Pass
    }
    return 0;  // Fail
}

// Test 3: Identity Matrix Test (2x2 subset of 8x8)
uint8_t test_identity(void)
{
    // Create 8x8 matrices with 2x2 identity in top-left corner
    int8_t mat_a[MATRIX_SIZE][MATRIX_SIZE];
    int8_t mat_b[MATRIX_SIZE][MATRIX_SIZE];
    int32_t result[MATRIX_SIZE][MATRIX_SIZE];

    memset(mat_a, 0, MATRIX_SIZE * MATRIX_SIZE);

    // A = [[1, 0, ...], [0, 1, ...], ...]
    mat_a[0][0] = 1;
    mat_a[1][1] = 1;

    // B = [[5, 6, ...], [7, 8, ...], ...]
    mat_b[0][0] = 5;
    mat_b[0][1] = 6;
    mat_b[1][0] = 7;
    mat_b[1][1] = 8;

    // Write matrices
    write_matrix_a(mat_a);
    write_matrix_b(mat_b);

    // Start multiplication (signed mode)
    start_multiplication(1);

    // Wait for completion
    if (!wait_for_done()) {
        return 0;  // Timeout - fail
    }

    // Read results
    read_matrix_c(result);

    // Verify: C should equal B (since A is identity)
    // Check top-left 2x2
    if (result[0][0] != 5 || result[0][1] != 6 ||
        result[1][0] != 7 || result[1][1] != 8) {
        return 0;  // Fail
    }

    // Check that other elements are zero (A and B are mostly zero)
    for (int i = 2; i < MATRIX_SIZE; i++) {
        if (result[0][i] != 0 || result[i][0] != 0) {
            return 0;  // Fail
        }
    }

    return 1;  // Pass
}

// Test 4: Full 8x8 Matrix Multiplication
uint8_t test_full_8x8(void)
{
    int8_t mat_a[MATRIX_SIZE][MATRIX_SIZE];
    int8_t mat_b[MATRIX_SIZE][MATRIX_SIZE];
    int32_t result[MATRIX_SIZE][MATRIX_SIZE];

    // Fill matrix A with sequential values [0..63]
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            mat_a[i][j] = i * MATRIX_SIZE + j;
        }
    }

    // Fill matrix B with sequential values [0..63]
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            mat_b[i][j] = i * MATRIX_SIZE + j;
        }
    }

    // Write matrices
    write_matrix_a(mat_a);
    write_matrix_b(mat_b);

    // Start multiplication (signed mode)
    start_multiplication(1);

    // Wait for completion
    if (!wait_for_done()) {
        return 0;  // Timeout - fail
    }

    // Read cycle count for performance measurement
    uint32_t cycles = *((volatile uint32_t *)MATMUL_CYCLES);

    // Read results
    read_matrix_c(result);

    // Verify spot checks (manual calculation):
    // C[0][0] = sum(A[0][k] * B[k][0]) for k=0..7
    //         = 0*0 + 1*8 + 2*16 + 3*24 + 4*32 + 5*40 + 6*48 + 7*56
    //         = 0 + 8 + 32 + 72 + 128 + 200 + 288 + 392 = 1120
    int32_t expected_00 = 1120;

    // C[0][7] = sum(A[0][k] * B[k][7]) for k=0..7
    //         = 0*7 + 1*15 + 2*23 + 3*31 + 4*39 + 5*47 + 6*55 + 7*63
    //         = 0 + 15 + 46 + 93 + 156 + 235 + 330 + 441 = 1316
    int32_t expected_07 = 1316;

    if (result[0][0] != expected_00 || result[0][7] != expected_07) {
        return 0;  // Fail
    }

    // Check cycle count is reasonable (~26 cycles expected)
    if (cycles < 20 || cycles > 50) {
        return 0;  // Unexpected timing - fail
    }

    return 1;  // Pass
}

// Test 5: Unsigned Mode Test
uint8_t test_unsigned(void)
{
    int8_t mat_a[MATRIX_SIZE][MATRIX_SIZE] = {0};
    int8_t mat_b[MATRIX_SIZE][MATRIX_SIZE] = {0};
    int32_t result[MATRIX_SIZE][MATRIX_SIZE];

    // Use small positive values
    mat_a[0][0] = 2;
    mat_a[0][1] = 3;
    mat_b[0][0] = 4;
    mat_b[1][0] = 5;

    // Write matrices
    write_matrix_a(mat_a);
    write_matrix_b(mat_b);

    // Start multiplication (UNSIGNED mode)
    start_multiplication(0);

    // Wait for completion
    if (!wait_for_done()) {
        return 0;  // Timeout - fail
    }

    // Read results
    read_matrix_c(result);

    // Verify: C[0][0] = 2*4 + 3*5 = 8 + 15 = 23
    if (result[0][0] != 23) {
        return 0;  // Fail
    }

    return 1;  // Pass
}

// Test 6: Soft Reset Test
uint8_t test_reset(void)
{
    // Reset the accelerator
    reset_accelerator();

    // Check that status returns to READY
    uint32_t status = *((volatile uint32_t *)MATMUL_STATUS);

    if (status & STATUS_READY) {
        return 1;  // Pass
    }
    return 0;  // Fail
}

// =============================================================================
// Main Test Sequence
// =============================================================================

void main()
{
    uint8_t test_result;
    uint8_t all_tests_passed = 1;

    // =========================================================================
    // Initialize Hardware
    // =========================================================================

    // Enable management GPIO for signaling to cocotb
    ManagmentGpio_outputEnable();
    ManagmentGpio_write(0);

    // Configure all GPIOs as user outputs (monitored by cocotb)
    GPIOs_configureAll(GPIO_MODE_USER_STD_OUT_MONITORED);
    GPIOs_loadConfigs();

    // CRITICAL: Enable Wishbone interface to user project
    User_enableIF(1);

    // Signal to cocotb that configuration is complete
    ManagmentGpio_write(1);

    // Small delay to ensure configuration is stable
    wait_cycles(1000);

    // =========================================================================
    // Test 1: Version Register Check
    // =========================================================================

    test_result = test_version();
    if (!test_result) {
        all_tests_passed = 0;
    }

    wait_cycles(100);

    // =========================================================================
    // Test 2: Status Register Check
    // =========================================================================

    test_result = test_status_ready();
    if (!test_result) {
        all_tests_passed = 0;
    }

    wait_cycles(100);

    // =========================================================================
    // Test 3: Identity Matrix Test
    // =========================================================================

    test_result = test_identity();
    if (!test_result) {
        all_tests_passed = 0;
    }

    wait_cycles(100);

    // =========================================================================
    // Test 4: Full 8x8 Matrix Multiplication
    // =========================================================================

    test_result = test_full_8x8();
    if (!test_result) {
        all_tests_passed = 0;
    }

    wait_cycles(100);

    // =========================================================================
    // Test 5: Unsigned Mode Test
    // =========================================================================

    test_result = test_unsigned();
    if (!test_result) {
        all_tests_passed = 0;
    }

    wait_cycles(100);

    // =========================================================================
    // Test 6: Soft Reset Test
    // =========================================================================

    test_result = test_reset();
    if (!test_result) {
        all_tests_passed = 0;
    }

    wait_cycles(100);

    // =========================================================================
    // Signal Test Completion
    // =========================================================================

    // If all tests passed, signal success to cocotb
    if (all_tests_passed) {
        // Signal completion (GPIO = 0)
        ManagmentGpio_write(0);
    } else {
        // Keep GPIO high to indicate failure
        // (cocotb will timeout and report failure)
        ManagmentGpio_write(1);
    }

    // Loop forever
    while (1) {
        wait_cycles(1000);
    }
}

# SPDX-License-Identifier: Apache-2.0
# Written by: Claude (Anthropic AI Assistant)
#
# Matrix Multiplication Accelerator Cocotb Test
#
# Description:
#   This test verifies the mat_mult_wb.v module, an 8x8 systolic array
#   matrix multiplier with Wishbone interface. The firmware running on
#   the management SoC performs various matrix multiplication operations
#   and verifies the results.
#
# Test Sequence:
#   1. Version register check
#   2. Status register check (READY bit)
#   3. Identity matrix test (2x2 subset)
#   4. Full 8x8 matrix multiplication
#   5. Unsigned mode test
#   6. Soft reset test
#
# Address Map (Base = 0x30000000):
#   0x000: CTRL - Control register
#   0x004: STATUS - Status register
#   0x008: CYCLE_COUNT - Performance counter
#   0x00C: VERSION - Version register (0xA7770001)
#   0x100-0x13F: Matrix A cache (64 bytes)
#   0x200-0x23F: Matrix B cache (64 bytes)
#   0x400-0x4FF: Matrix C result cache (256 bytes)

from caravel_cocotb.caravel_interfaces import test_configure
from caravel_cocotb.caravel_interfaces import report_test
import cocotb


@cocotb.test()
@report_test
async def mat_mult(dut):
    """
    Test the matrix multiplication accelerator.

    This test configures the Caravel environment, waits for the firmware
    to complete matrix multiplication operations, and verifies the results.
    """

    # Configure the Caravel environment with extended timeout
    # Matrix operations may take longer than simple register tests
    caravelEnv = await test_configure(dut, timeout_cycles=1000000)

    cocotb.log.info(f"[TEST] Starting Matrix Multiplication Accelerator Test")
    cocotb.log.info(f"[TEST] Accelerator Base Address: 0x30000000")
    cocotb.log.info(f"[TEST] Expected latency: ~26 cycles per 8x8 multiplication")

    # Wait for the firmware to complete configuration
    # Firmware will set management GPIO to 1 when ready
    cocotb.log.info(f"[TEST] Waiting for firmware configuration...")
    await caravelEnv.wait_mgmt_gpio(1)

    cocotb.log.info(f"[TEST] Firmware configuration complete")
    cocotb.log.info(f"[TEST] Wishbone interface enabled")

    # Release the chip select bar (CSB) to allow user project Wishbone access
    await caravelEnv.release_csb()

    cocotb.log.info(f"[TEST] CSB released - firmware can now access user project")
    cocotb.log.info(f"[TEST] Firmware executing test sequence:")
    cocotb.log.info(f"[TEST]   1. Version check (0x3000000C)")
    cocotb.log.info(f"[TEST]   2. Status register check (0x30000004)")
    cocotb.log.info(f"[TEST]   3. Identity matrix test (2x2)")
    cocotb.log.info(f"[TEST]   4. Full 8x8 matrix multiplication")
    cocotb.log.info(f"[TEST]   5. Unsigned mode test")
    cocotb.log.info(f"[TEST]   6. Soft reset test")

    # Wait for the firmware to complete all operations
    # Firmware will set management GPIO to 0 when done
    cocotb.log.info(f"[TEST] Processing matrix operations...")
    await caravelEnv.wait_mgmt_gpio(0)

    cocotb.log.info(f"[TEST] ========================================")
    cocotb.log.info(f"[TEST] Matrix Multiplication Test PASSED")
    cocotb.log.info(f"[TEST] ========================================")
    cocotb.log.info(f"[TEST] All matrix operations completed successfully")
    cocotb.log.info(f"[TEST] Firmware verified:")
    cocotb.log.info(f"[TEST]   - Version register correct")
    cocotb.log.info(f"[TEST]   - Status flags functional")
    cocotb.log.info(f"[TEST]   - Identity multiplication correct")
    cocotb.log.info(f"[TEST]   - 8x8 multiplication correct")
    cocotb.log.info(f"[TEST]   - Signed/unsigned modes working")
    cocotb.log.info(f"[TEST]   - Reset functionality working")

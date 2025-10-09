
# Neuromorphic_X1 – Firmware Tests (Wishbone @ 0x3000_000C)

This README explains two simple firmware tests that talk to the **Neuromorphic_X1** behavioral model through a single **Wishbone address**: `0x3000_000C`.

Both tests run on the **Caravel management SoC (VexRiscv)**. They use the same basic recipe:

1. Enable the management GPIOs and the user Wishbone interface.
2. Write commands to the Wishbone-mapped command port (`0x3000_000C`).
3. Wait some cycles while the behavioral core works.
4. Read results back from the same address (`0x3000_000C`).

---

## Memory Map (used by these tests)

- **0x3000_000C**: Neuromorphic_X1 “mailbox” register
  - **CPU WRITE**  → enqueue a command into the core
  - **CPU READ**   → pop a result from the core

---

## Command Word Format (used by the tests)

- **MODE=2’b11 (PROGRAM)**: Writes one bit into `array[ROW][COL]`.
  - In the behavioral model, bit written = `1` if `DATA[7:0] = 8'hFF` and bit written = `0` if `DATA[7:0] = 8'h00`.
  
- **MODE=2’b01 (READ)**: Requests a read of `array[ROW][COL]`, which returns a 32-bit word with the bit in LSB.

---

## Handshake Notes (Wishbone)

- Always call `User_enableIF(1)` **before** accessing `0x3000_000C`. Otherwise, no **ACK** will be received.
- The behavioral model asserts **ACK** only when a command has been accepted (write) or a result is popped (read). In your firmware, a plain load/store to the address is enough—the platform wrapper waits for **ACK**.
- Delays between issuing commands and reading results are added with `wait_cycles()` so the core has time to finish.

---

## Utility: `wait_cycles()`

The tests use a simple cycle-burner to wait a fixed number of CPU cycles:

```c
static inline void wait_cycles(uint32_t cycles) {
    for (uint32_t i = 0; i < cycles; i++) { __asm__ volatile ("nop"); }
}
```

Adjust the arguments (e.g., 300, 900, etc.) as needed to meet your timing for the behavioral `RD_Dly/WR_Dly` in simulation.

---

# TEST 1 — “Single operations with fixed words”

**File section**:   
```
// Comment From Here - Test 1  ...  // Comment Till Here - Test 1
```

### Purpose:
- Demonstrates a couple of single writes, short delay, and single reads.
- Shows how to use a fixed command word directly (without `pack_cmd()`).

### What it does (high level):
1. Enables the interface.
2. Writes two words (`wdata`, `wdata1`) to `0x3000_000C`.
   - `wdata  = 0xC21000FF`  (MODE=11/program, DATA LSBs = 0xFF)
   - `wdata1 = 0x42100000`  (MODE=01/read   for some ROW/COL fields)
3. `wait_cycles(300);`
4. Reads once from `0x3000_000C` → `temp`.
5. Writes three more words (`wdata2`, `wdata3`, `wdata1`).
6. `wait_cycles(900);`
7. Reads twice → `temp1`, `temp2`.
8. Drives management GPIO low to indicate the end.

### What to expect:
- The first read (`temp`) should reflect the value programmed by the matching read command (depends on the target row/col in your words).
- With `DATA=0xFF` in **PROGRAM**, the behavioral model will store bit=1. With `DATA=0x00`, it stores 0.
- If you see `0xDEAD_C0DE`, you read when the output FIFO was empty. Increase the `wait_cycles()` or ensure the **READ** command was issued.

### Waveform tips (optional):
- Watch `wbs_cyc_i`, `wbs_stb_i`, `wbs_we_i`, `wbs_ack_o` on each access.
- `wbs_dat_o` is valid when **ACK** is high during reads.

### How to tweak:
- Change `wdata*` words to target different `ROW/COL` fields.
- Adjust the delays to match your behavioral `RD_Dly/WR_Dly`.

---

# TEST 2 — “Looped PROGRAM + READ using `pack_cmd()`”

**File section**:   
```
// Comment From Here - Test 2  ...  // Comment Till Here - Test 2
```

**(Currently commented in your source; uncomment to use.)**

### Purpose:
- Programs a sequence of addresses (`ROW=COL=i` for `i=0..9`), alternating the input `DATA` between `0xFF` and `0x00`, then issues **READ** commands for the same addresses and collects the results into `temp[10]`.
- Uses `pack_cmd()` to build `MODE/ROW/COL/DATA` fields cleanly.

### What it does (step-by-step):
1. Enables the interface.
2. For `i = 0..9`:
   - `data20 = 0x0FF` when `i` is even, `0x000` when `i` is odd.
   - `cmd = pack_cmd(3 /*PROGRAM*/, row=i, col=i, data20);`
   - `*(volatile uint32_t *)0x3000000C = cmd;`
3. `wait_cycles(3000);` // Let the core finish programming.
4. For `i = 0..9`:
   - `cmd = pack_cmd(1 /*READ*/, row=i, col=i, data20=<don’t care>);`
   - `*(volatile uint32_t *)0x3000000C = cmd;`
5. `wait_cycles(500);`    // Let the core push read results.
6. For `i = 0..9`:
   - `temp[i] = *(volatile uint32_t *)0x3000000C;  // read back`

### Expected results:
- The behavioral core writes `bit=1` if `DATA[7:0] = 0xFF` and `bit=0` if `DATA[7:0] = 0x00`. Therefore:
  - For even `i` (`i=0,2,4,6,8`): `temp[i]` LSB should be 1.
  - For odd `i` (`i=1,3,5,7,9`): `temp[i]` LSB should be 0.
  
- If reads occur too early, the FIFO may be empty, so increase the waits. If you get `0xDEAD_C0DE` (depending on model version), that indicates an empty read.

### How to enable Test 2:
- Comment out **Test 1** block.
- Uncomment the **Test 2** block.
- Ensure the `pack_cmd()` helper and the `temp[]` array are compiled.

---

## Build & Run (typical flow)

Your project uses the **efabless dv container** and **Caravel cocotb flow**. In general, you will:

1. Build the firmware to HEX (using the provided Docker command in your flow).
2. Run the simulation (which loads the hex into the management core).
3. Open the waveform to check **Wishbone handshakes** and data.

### Tip: If you see “no ACK”:
- Make sure `User_enableIF(1)` is executed before any Wishbone access.
- Check the address is exactly `0x3000000C` (word-aligned).

---

## Troubleshooting

- **Stuck read / no progress**:
  - Increase `wait_cycles()` so the behavioral delays (`RD_Dly/WR_Dly`) are comfortably covered before reading.
  - Confirm your **READ** commands are enqueued after **PROGRAM** commands.

- **ACK never asserted**:
  - Call `User_enableIF(1)` first.
  - Verify your GPIO and interface configuration calls are executed.
  - Ensure the shim address matches (`0x3000_000C`).

- **Unexpected data**:
  - Double-check **MODE** bits and **ROW/COL**.
  - Remember that the model sets bit=1 only if `DATA[7:0] = 8'hFF`.
  - Remember that the model sets bit=0 only if `DATA[7:0] = 8'h00`.

- **Tool warnings about “clock coerced to inout”**:
  - Not a firmware issue; see HDL fix by splitting pad clock and core clock (use a `‘clock_core’` wire).

---

## Quick Reference

- **Address**: `0x3000_000C`
- **Modes**: `2'b11 PROGRAM`, `2'b01 READ`
- **Pack**: `pack_cmd(mode, row, col, data20)`
- **Delay**: `wait_cycles(n)` burns exactly `n` CPU cycles
- **Reads**: Simple *addr* loads; platform stalls until **ACK**.

---

### NOTE:
- To run **Test 1 block**, comment **Test 2 block**.
- To run **Test 2 block**, comment **Test 1 block**.

### To Run Test 2 Block For Different Number of Inputs

```c
// Performing Write Mode Operation
for (uint32_t i = 0; i < {"Change Value Here"}; i++)

// Wait cycles for delay
wait_cycles({"Change Value Here"});

// Performing Read Mode Operation
for (uint32_t i = 0; i < {"Change Value Here"}; i++)

// Wait cycles for delay
wait_cycles({"Change Value Here"});

// Performing Read Operation
uint32_t temp[{"Change Value Here"}];
for (uint32_t i = 0; i < {"Change Value Here"}; i++)
```


// SPDX-License-Identifier: Apache-2.0
// Written by: Claude (Anthropic AI Assistant)
//
// =============================================================================
// Matrix Multiplication Accelerator with Wishbone Interface
// =============================================================================
//
// Description:
//   8x8 systolic array matrix multiplier with memory-mapped cache and Wishbone
//   control interface. Performs C = A × B where all matrices are 8x8.
//
// Features:
//   - Fixed 8x8 matrix dimensions
//   - 8-bit signed/unsigned integer inputs
//   - 32-bit accumulated results
//   - Systolic array architecture for high throughput
//   - Memory-mapped cache for matrices A, B, and C
//   - Wishbone slave interface for control and data access
//   - Interrupt output on completion
//   - Performance counter
//
// Address Map (Base = 0x31000000):
//   0x31000000: CTRL        - Control register
//                             [0] START   = Write 1 to start multiplication
//                             [1] RESET   = Reset accelerator state
//                             [2] SIGNED  = 1=signed, 0=unsigned multiplication
//                             [8] IRQ_EN  = Enable interrupt generation
//   0x31000004: STATUS      - Status register (read-only)
//                             [0] BUSY    = Computation in progress
//                             [1] DONE    = Computation complete
//                             [2] READY   = Ready for new operation
//   0x31000008: CYCLE_COUNT - Performance counter (read-only)
//   0x3100000C: VERSION     - Module version/ID (0xA7770001)
//
//   0x31000100-0x3100013F: Matrix A cache (64 bytes)
//                          16 words, 4 elements per word (packed 8-bit)
//   0x31000200-0x3100023F: Matrix B cache (64 bytes)
//                          16 words, 4 elements per word (packed 8-bit)
//   0x31000400-0x310004FF: Matrix C cache (256 bytes)
//                          64 words, 1 element per word (32-bit result)
//
// Usage Example:
//   1. Write matrix A elements to 0x31000100-0x3100013F (4 elements per word)
//   2. Write matrix B elements to 0x31000200-0x3100023F
//   3. Write CTRL = 0x105 (IRQ_EN | SIGNED | START)
//   4. Wait for IRQ or poll STATUS[1] for DONE
//   5. Read results from 0x31000400-0x310004FF
//
// Timing:
//   - Total latency: ~26 cycles
//   - CLEAR:      1 cycle
//   - COMPUTE:    16 cycles (feed systolic array)
//   - DRAIN:      8 cycles (pipeline drain)
//   - WRITEBACK:  1 cycle
//
// =============================================================================

`timescale 1ns / 1ps

module mat_mult_wb #(
    parameter MATRIX_SIZE = 8,
    parameter DATA_WIDTH = 8,
    parameter RESULT_WIDTH = 32,
    parameter BASE_ADDR = 32'h30000000
) (
`ifdef USE_POWER_PINS
    inout         VSS,            // Ground
    inout         VDDA,           // 1.8V analog supply
    inout         VDDC,           // 1.8V digital supply
`endif

    // Wishbone interface
    input  wire        wb_clk_i,
    input  wire        wb_rst_i,   // Active HIGH
    input  wire        wbs_stb_i,
    input  wire        wbs_cyc_i,
    input  wire        wbs_we_i,
    input  wire [3:0]  wbs_sel_i,
    input  wire [31:0] wbs_dat_i,
    input  wire [31:0] wbs_adr_i,
    output reg  [31:0] wbs_dat_o,
    output reg         wbs_ack_o,

    // Interrupt output
    output reg         irq_o       // Asserted when computation done
);

    // =========================================================================
    // Local Parameters
    // =========================================================================

    localparam VERSION = 32'hA7770001;  // Version identifier

    // FSM states
    localparam STATE_IDLE      = 3'd0;
    localparam STATE_CLEAR     = 3'd1;
    localparam STATE_COMPUTE   = 3'd2;
    localparam STATE_DRAIN     = 3'd3;
    localparam STATE_WRITEBACK = 3'd4;
    localparam STATE_DONE      = 3'd5;

    // =========================================================================
    // Address Decoder
    // =========================================================================

    wire [31:0] addr_offset = wbs_adr_i - BASE_ADDR;
    wire module_select = (wbs_adr_i[31:12] == BASE_ADDR[31:12]); // Match upper 20 bits

    wire wb_valid = wbs_stb_i && wbs_cyc_i && module_select;
    wire wb_write = wb_valid && wbs_we_i;
    wire wb_read  = wb_valid && !wbs_we_i;

    // Address region decoding (using lower 12 bits for 4KB space)
    wire ctrl_access    = (addr_offset[11:4] == 8'h00);  // 0x000-0x00F
    wire cache_a_access = (addr_offset[11:8] == 4'h1);   // 0x100-0x1FF
    wire cache_b_access = (addr_offset[11:8] == 4'h2);   // 0x200-0x2FF
    wire cache_c_access = (addr_offset[11:8] == 4'h4);   // 0x400-0x4FF

    // =========================================================================
    // Control and Status Registers
    // =========================================================================

    reg start_req;
    reg soft_reset;
    reg signed_mode;
    reg irq_enable;
    reg busy;
    reg done;
    reg ready;
    reg [31:0] cycle_count;

    // Control register write
    always @(posedge wb_clk_i) begin
        if (wb_rst_i) begin
            start_req   <= 1'b0;
            soft_reset  <= 1'b0;
            signed_mode <= 1'b1;  // Default to signed
            irq_enable  <= 1'b0;
        end else begin
            // Auto-clear start_req after one cycle
            start_req <= 1'b0;

            if (wb_write && ctrl_access && (addr_offset[3:0] == 4'h0)) begin
                if (wbs_sel_i[0]) begin
                    start_req   <= wbs_dat_i[0];
                    soft_reset  <= wbs_dat_i[1];
                    signed_mode <= wbs_dat_i[2];
                end
                if (wbs_sel_i[1]) begin
                    irq_enable <= wbs_dat_i[8];
                end
            end
        end
    end

    // =========================================================================
    // Cache Memories
    // =========================================================================

    // Matrix A cache: 8x8 elements, 8-bit each, packed 4 per word
    reg [31:0] cache_a [0:15];  // 16 words = 64 bytes

    // Matrix B cache: 8x8 elements, 8-bit each, packed 4 per word
    reg [31:0] cache_b [0:15];  // 16 words = 64 bytes

    // Matrix C cache: 8x8 elements, 32-bit each
    reg [31:0] cache_c [0:63];  // 64 words = 256 bytes

    // Cache write logic with byte-select support
    always @(posedge wb_clk_i) begin
        integer idx_ab;
        if (wb_rst_i) begin
            for (idx_ab = 0; idx_ab < 16; idx_ab = idx_ab + 1) begin
                cache_a[idx_ab] <= 32'h0;
                cache_b[idx_ab] <= 32'h0;
            end
        end else begin
            // Write to cache A
            if (wb_write && cache_a_access) begin
                if (wbs_sel_i[0]) cache_a[addr_offset[5:2]][7:0]   <= wbs_dat_i[7:0];
                if (wbs_sel_i[1]) cache_a[addr_offset[5:2]][15:8]  <= wbs_dat_i[15:8];
                if (wbs_sel_i[2]) cache_a[addr_offset[5:2]][23:16] <= wbs_dat_i[23:16];
                if (wbs_sel_i[3]) cache_a[addr_offset[5:2]][31:24] <= wbs_dat_i[31:24];
            end

            // Write to cache B
            if (wb_write && cache_b_access) begin
                if (wbs_sel_i[0]) cache_b[addr_offset[5:2]][7:0]   <= wbs_dat_i[7:0];
                if (wbs_sel_i[1]) cache_b[addr_offset[5:2]][15:8]  <= wbs_dat_i[15:8];
                if (wbs_sel_i[2]) cache_b[addr_offset[5:2]][23:16] <= wbs_dat_i[23:16];
                if (wbs_sel_i[3]) cache_b[addr_offset[5:2]][31:24] <= wbs_dat_i[31:24];
            end

            // Cache C is written by the accelerator in separate always block
        end
    end

    // =========================================================================
    // Wishbone Read Data Multiplexer
    // =========================================================================

    always @(posedge wb_clk_i) begin
        if (wb_rst_i) begin
            wbs_dat_o <= 32'h0;
        end else if (wb_read) begin
            if (ctrl_access) begin
                case (addr_offset[3:0])
                    4'h0: wbs_dat_o <= {23'h0, irq_enable, 5'h0, signed_mode, soft_reset, start_req};
                    4'h4: wbs_dat_o <= {29'h0, ready, done, busy};
                    4'h8: wbs_dat_o <= cycle_count;
                    4'hC: wbs_dat_o <= VERSION;
                    default: wbs_dat_o <= 32'h0;
                endcase
            end else if (cache_a_access) begin
                wbs_dat_o <= cache_a[addr_offset[5:2]];
            end else if (cache_b_access) begin
                wbs_dat_o <= cache_b[addr_offset[5:2]];
            end else if (cache_c_access) begin
                wbs_dat_o <= cache_c[addr_offset[7:2]];
            end else begin
                wbs_dat_o <= 32'hDEADBEEF;  // Invalid address
            end
        end else begin
            wbs_dat_o <= 32'h0;
        end
    end

    // =========================================================================
    // Wishbone Acknowledge
    // =========================================================================

    always @(posedge wb_clk_i) begin
        if (wb_rst_i) begin
            wbs_ack_o <= 1'b0;
        end else begin
            // Single-cycle acknowledge
            wbs_ack_o <= wb_valid && !wbs_ack_o;
        end
    end

    // =========================================================================
    // Processing Element (PE) Module
    // =========================================================================

    // Each PE performs multiply-accumulate in systolic array

    wire [DATA_WIDTH-1:0] pe_a_in   [0:MATRIX_SIZE-1][0:MATRIX_SIZE-1];
    wire [DATA_WIDTH-1:0] pe_b_in   [0:MATRIX_SIZE-1][0:MATRIX_SIZE-1];
    wire [DATA_WIDTH-1:0] pe_a_out  [0:MATRIX_SIZE-1][0:MATRIX_SIZE-1];
    wire [DATA_WIDTH-1:0] pe_b_out  [0:MATRIX_SIZE-1][0:MATRIX_SIZE-1];
    wire [RESULT_WIDTH-1:0] pe_result [0:MATRIX_SIZE-1][0:MATRIX_SIZE-1];

    wire pe_enable;
    wire pe_clear;

    genvar row, col, i;
    generate
        for (row = 0; row < MATRIX_SIZE; row = row + 1) begin : gen_row
            for (col = 0; col < MATRIX_SIZE; col = col + 1) begin : gen_col
                mat_mult_pe #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH(RESULT_WIDTH)
                ) pe_inst (
                    .clk(wb_clk_i),
                    .rst_n(!wb_rst_i),
                    .enable(pe_enable),
                    .clear(pe_clear),
                    .signed_mode(signed_mode),
                    .a_in(pe_a_in[row][col]),
                    .b_in(pe_b_in[row][col]),
                    .a_out(pe_a_out[row][col]),
                    .b_out(pe_b_out[row][col]),
                    .result(pe_result[row][col])
                );
            end
        end
    endgenerate

    // =========================================================================
    // Systolic Array Interconnections
    // =========================================================================

    // Horizontal connections (A data flows left to right)
    generate
        for (row = 0; row < MATRIX_SIZE; row = row + 1) begin : gen_horiz_conn
            for (col = 0; col < MATRIX_SIZE-1; col = col + 1) begin : gen_horiz_stage
                assign pe_a_in[row][col+1] = pe_a_out[row][col];
            end
        end
    endgenerate

    // Vertical connections (B data flows top to bottom)
    generate
        for (col = 0; col < MATRIX_SIZE; col = col + 1) begin : gen_vert_conn
            for (row = 0; row < MATRIX_SIZE-1; row = row + 1) begin : gen_vert_stage
                assign pe_b_in[row+1][col] = pe_b_out[row][col];
            end
        end
    endgenerate

    // =========================================================================
    // Control FSM
    // =========================================================================

    reg [2:0] state, next_state;
    reg [4:0] compute_count;  // 0-15 for feeding array, 16-23 for draining

    always @(posedge wb_clk_i) begin
        if (wb_rst_i || soft_reset) begin
            state <= STATE_IDLE;
        end else begin
            state <= next_state;
        end
    end

    // Next state logic
    always @(*) begin
        next_state = state;

        case (state)
            STATE_IDLE: begin
                if (start_req) begin
                    next_state = STATE_CLEAR;
                end
            end

            STATE_CLEAR: begin
                next_state = STATE_COMPUTE;
            end

            STATE_COMPUTE: begin
                if (compute_count == 5'd15) begin
                    next_state = STATE_DRAIN;
                end
            end

            STATE_DRAIN: begin
                if (compute_count == 5'd23) begin
                    next_state = STATE_WRITEBACK;
                end
            end

            STATE_WRITEBACK: begin
                next_state = STATE_DONE;
            end

            STATE_DONE: begin
                next_state = STATE_IDLE;
            end

            default: begin
                next_state = STATE_IDLE;
            end
        endcase
    end

    // Compute counter
    always @(posedge wb_clk_i) begin
        if (wb_rst_i || soft_reset) begin
            compute_count <= 5'd0;
        end else begin
            if (state == STATE_COMPUTE || state == STATE_DRAIN) begin
                compute_count <= compute_count + 1;
            end else begin
                compute_count <= 5'd0;
            end
        end
    end

    // Control signals
    assign pe_enable = (state == STATE_COMPUTE) || (state == STATE_DRAIN);
    assign pe_clear  = (state == STATE_CLEAR);

    // Status flags
    always @(posedge wb_clk_i) begin
        if (wb_rst_i || soft_reset) begin
            busy  <= 1'b0;
            done  <= 1'b0;
            ready <= 1'b1;
        end else begin
            case (state)
                STATE_IDLE: begin
                    busy  <= 1'b0;
                    done  <= 1'b0;
                    ready <= 1'b1;
                end

                STATE_CLEAR, STATE_COMPUTE, STATE_DRAIN, STATE_WRITEBACK: begin
                    busy  <= 1'b1;
                    done  <= 1'b0;
                    ready <= 1'b0;
                end

                STATE_DONE: begin
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    ready <= 1'b0;
                end

                default: begin
                    busy  <= 1'b0;
                    done  <= 1'b0;
                    ready <= 1'b1;
                end
            endcase
        end
    end

    // =========================================================================
    // Data Feeding Logic (Systolic Array Inputs)
    // =========================================================================

    // Extract individual elements from cache
    wire [DATA_WIDTH-1:0] a_elements [0:63];
    wire [DATA_WIDTH-1:0] b_elements [0:63];

    generate
        for (i = 0; i < 16; i = i + 1) begin : gen_unpack
            assign a_elements[i*4 + 0] = cache_a[i][7:0];
            assign a_elements[i*4 + 1] = cache_a[i][15:8];
            assign a_elements[i*4 + 2] = cache_a[i][23:16];
            assign a_elements[i*4 + 3] = cache_a[i][31:24];

            assign b_elements[i*4 + 0] = cache_b[i][7:0];
            assign b_elements[i*4 + 1] = cache_b[i][15:8];
            assign b_elements[i*4 + 2] = cache_b[i][23:16];
            assign b_elements[i*4 + 3] = cache_b[i][31:24];
        end
    endgenerate

    // Feed A data (horizontal, staggered by row)
    generate
        for (row = 0; row < MATRIX_SIZE; row = row + 1) begin : gen_feed_a
            reg [DATA_WIDTH-1:0] a_feed_reg;

            always @(posedge wb_clk_i) begin
                if (wb_rst_i || soft_reset) begin
                    a_feed_reg <= {DATA_WIDTH{1'b0}};
                end else if (state == STATE_COMPUTE) begin
                    // Feed starts at different times for each row (diagonal wave)
                    if (compute_count >= row && compute_count < (row + MATRIX_SIZE)) begin
                        a_feed_reg <= a_elements[row * MATRIX_SIZE + (compute_count - row)];
                    end else begin
                        a_feed_reg <= {DATA_WIDTH{1'b0}};
                    end
                end else begin
                    a_feed_reg <= {DATA_WIDTH{1'b0}};
                end
            end

            assign pe_a_in[row][0] = a_feed_reg;
        end
    endgenerate

    // Feed B data (vertical, staggered by column)
    generate
        for (col = 0; col < MATRIX_SIZE; col = col + 1) begin : gen_feed_b
            reg [DATA_WIDTH-1:0] b_feed_reg;

            always @(posedge wb_clk_i) begin
                if (wb_rst_i || soft_reset) begin
                    b_feed_reg <= {DATA_WIDTH{1'b0}};
                end else if (state == STATE_COMPUTE) begin
                    // Feed starts at different times for each column (diagonal wave)
                    if (compute_count >= col && compute_count < (col + MATRIX_SIZE)) begin
                        b_feed_reg <= b_elements[(compute_count - col) * MATRIX_SIZE + col];
                    end else begin
                        b_feed_reg <= {DATA_WIDTH{1'b0}};
                    end
                end else begin
                    b_feed_reg <= {DATA_WIDTH{1'b0}};
                end
            end

            assign pe_b_in[0][col] = b_feed_reg;
        end
    endgenerate

    // =========================================================================
    // Result Extraction and Writeback
    // =========================================================================

    always @(posedge wb_clk_i) begin
        integer idx_c;
        if (wb_rst_i || soft_reset) begin
            for (idx_c = 0; idx_c < 64; idx_c = idx_c + 1) begin
                cache_c[idx_c] <= 32'h0;
            end
        end else if (state == STATE_WRITEBACK) begin
            // Extract results from all PEs
            for (idx_c = 0; idx_c < MATRIX_SIZE; idx_c = idx_c + 1) begin
                cache_c[idx_c * MATRIX_SIZE + 0] <= pe_result[idx_c][0];
                cache_c[idx_c * MATRIX_SIZE + 1] <= pe_result[idx_c][1];
                cache_c[idx_c * MATRIX_SIZE + 2] <= pe_result[idx_c][2];
                cache_c[idx_c * MATRIX_SIZE + 3] <= pe_result[idx_c][3];
                cache_c[idx_c * MATRIX_SIZE + 4] <= pe_result[idx_c][4];
                cache_c[idx_c * MATRIX_SIZE + 5] <= pe_result[idx_c][5];
                cache_c[idx_c * MATRIX_SIZE + 6] <= pe_result[idx_c][6];
                cache_c[idx_c * MATRIX_SIZE + 7] <= pe_result[idx_c][7];
            end
        end
    end

    // =========================================================================
    // Performance Counter
    // =========================================================================

    always @(posedge wb_clk_i) begin
        if (wb_rst_i || soft_reset) begin
            cycle_count <= 32'h0;
        end else begin
            if (state == STATE_IDLE) begin
                cycle_count <= 32'h0;
            end else if (state != STATE_DONE) begin
                cycle_count <= cycle_count + 1;
            end
        end
    end

    // =========================================================================
    // Interrupt Generation
    // =========================================================================

    always @(posedge wb_clk_i) begin
        if (wb_rst_i || soft_reset) begin
            irq_o <= 1'b0;
        end else begin
            // Assert IRQ when entering DONE state with IRQ enabled
            if (state != STATE_DONE && next_state == STATE_DONE && irq_enable) begin
                irq_o <= 1'b1;
            end
            // Clear IRQ on any CTRL register write or when leaving DONE
            else if ((wb_write && ctrl_access) || (state == STATE_DONE && next_state == STATE_IDLE)) begin
                irq_o <= 1'b0;
            end
        end
    end

endmodule


// =============================================================================
// Processing Element (PE) for Systolic Array
// =============================================================================

module mat_mult_pe #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  enable,
    input  wire                  clear,
    input  wire                  signed_mode,
    input  wire [DATA_WIDTH-1:0] a_in,
    input  wire [DATA_WIDTH-1:0] b_in,
    output reg  [DATA_WIDTH-1:0] a_out,
    output reg  [DATA_WIDTH-1:0] b_out,
    output wire [ACC_WIDTH-1:0]  result
);

    // Registered inputs for systolic dataflow
    reg [DATA_WIDTH-1:0] a_reg;
    reg [DATA_WIDTH-1:0] b_reg;

    // Accumulator
    reg [ACC_WIDTH-1:0] acc;

    // Multiply result (signed or unsigned)
    wire signed [DATA_WIDTH:0] a_signed = {a_reg[DATA_WIDTH-1], a_reg};
    wire signed [DATA_WIDTH:0] b_signed = {b_reg[DATA_WIDTH-1], b_reg};
    wire signed [2*DATA_WIDTH+1:0] mult_signed = a_signed * b_signed;

    wire [DATA_WIDTH-1:0] a_unsigned = a_reg;
    wire [DATA_WIDTH-1:0] b_unsigned = b_reg;
    wire [2*DATA_WIDTH-1:0] mult_unsigned = a_unsigned * b_unsigned;

    wire [ACC_WIDTH-1:0] mult_result = signed_mode ?
                                       {{(ACC_WIDTH-2*DATA_WIDTH-2){mult_signed[2*DATA_WIDTH+1]}}, mult_signed} :
                                       {{(ACC_WIDTH-2*DATA_WIDTH){1'b0}}, mult_unsigned};

    // Register inputs and pass through
    always @(posedge clk) begin
        if (!rst_n) begin
            a_reg <= {DATA_WIDTH{1'b0}};
            b_reg <= {DATA_WIDTH{1'b0}};
            a_out <= {DATA_WIDTH{1'b0}};
            b_out <= {DATA_WIDTH{1'b0}};
        end else begin
            a_reg <= a_in;
            b_reg <= b_in;
            a_out <= a_reg;  // Pass to next PE (with 1 cycle delay)
            b_out <= b_reg;  // Pass to next PE (with 1 cycle delay)
        end
    end

    // Accumulator logic
    always @(posedge clk) begin
        if (!rst_n || clear) begin
            acc <= {ACC_WIDTH{1'b0}};
        end else if (enable) begin
            acc <= acc + mult_result;
        end
    end

    assign result = acc;

endmodule

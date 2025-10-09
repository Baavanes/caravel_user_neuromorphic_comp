`default_nettype none
module user_project_wrapper #(
    parameter BITS = 32
) (
`ifdef USE_POWER_PINS
    inout vdda1, inout vdda2,
    inout vssa1, inout vssa2,
    inout vccd1, inout vccd2,
    inout vssd1, inout vssd2,
`endif

    // Wishbone
    input         wb_clk_i,
    input         wb_rst_i,
    input         wbs_stb_i,
    input         wbs_cyc_i,
    input         wbs_we_i,
    input  [3:0]  wbs_sel_i,
    input  [31:0] wbs_dat_i,
    input  [31:0] wbs_adr_i,
    output        wbs_ack_o,
    output [31:0] wbs_dat_o,

    // Logic Analyzer
    input  [127:0] la_data_in,
    output [127:0] la_data_out,
    input  [127:0] la_oenb,

    // Digital IOs
    input  [`MPRJ_IO_PADS-1:0] io_in,
    output [`MPRJ_IO_PADS-1:0] io_out,
    output [`MPRJ_IO_PADS-1:0] io_oeb,

    // Analog IOs (analog_io[k] <-> GPIO pad k+7)
    inout  [`MPRJ_IO_PADS-10:0] analog_io,

    // Extra user clock
    input   user_clock2,

    // IRQs
    output [2:0] user_irq
);

    // ------------------------------------------------------------
    // Wishbone address map for two slaves (4 KB windows each)
    // ------------------------------------------------------------
    localparam [31:0] MPRJ0_BASE = 32'h0000_0000;
    localparam [31:0] MPRJ1_BASE = 32'h0000_1000;
    localparam [31:0] MPRJ_MASK  = 32'hFFFF_F000; // 4KB region

    wire sel_mprj0 = ((wbs_adr_i & MPRJ_MASK) == MPRJ0_BASE);
    wire sel_mprj1 = ((wbs_adr_i & MPRJ_MASK) == MPRJ1_BASE);

    // Gate cyc/stb so only the selected slave is active
    wire wbs_cyc_i_0 = wbs_cyc_i & sel_mprj0;
    wire wbs_stb_i_0 = wbs_stb_i & sel_mprj0;

    wire wbs_cyc_i_1 = wbs_cyc_i & sel_mprj1;
    wire wbs_stb_i_1 = wbs_stb_i & sel_mprj1;

    // Return paths from each slave
    wire        wbs_ack_o_0, wbs_ack_o_1;
    wire [31:0] wbs_dat_o_0, wbs_dat_o_1;

    // ------------------------------------------------------------
    // Instance 0 (mprj0)
    // ------------------------------------------------------------
    Neuromorphic_X1_wb mprj0 (
    `ifdef USE_POWER_PINS
        .VDDC (vccd1),
        .VDDA (vdda1),
        .VSS  (vssd1),
    `endif

        // Clocks / resets
        .user_clk (wb_clk_i),
        .user_rst (wb_rst_i),
        .wb_clk_i (wb_clk_i),
        .wb_rst_i (wb_rst_i),

        // Wishbone (gated)
        .wbs_stb_i (wbs_stb_i_0),
        .wbs_cyc_i (wbs_cyc_i_0),
        .wbs_we_i  (wbs_we_i),
        .wbs_sel_i (wbs_sel_i),
        .wbs_dat_i (wbs_dat_i),
        .wbs_adr_i (wbs_adr_i),
        .wbs_dat_o (wbs_dat_o_0),
        .wbs_ack_o (wbs_ack_o_0),

  // Scan/Test
  .ScanInCC  (io_in[4]),
  .ScanInDL  (io_in[1]),
  .ScanInDR  (io_in[2]),
  .TM        (io_in[5]),
  .ScanOutCC (io_out[0]),

        // Analog / bias pins
        // NOTE: Currently shared with instance 1. If you need distinct biasing,
        //       assign a different analog_io range to mprj1 below.
        .Iref          (analog_io[0]),
        .Vcc_read      (analog_io[1]),
        .Vcomp         (analog_io[2]),
        .Bias_comp2    (analog_io[3]),
        .Vcc_wl_read   (analog_io[12]),
        .Vcc_wl_set    (analog_io[5]),
        .Vbias         (analog_io[6]),
        .Vcc_wl_reset  (analog_io[7]),
        .Vcc_set       (analog_io[8]),
        .Vcc_reset     (analog_io[9]),
        .Vcc_L         (analog_io[10]),
        .Vcc_Body      (analog_io[11])
    );

    // ------------------------------------------------------------
    // Instance 1 (mprj1)
    // ------------------------------------------------------------
    Neuromorphic_X1_wb mprj1 (
    `ifdef USE_POWER_PINS
        .VDDC (vccd1),
        .VDDA (vdda1),
        .VSS  (vssd1),
    `endif

        // Clocks / resets
        .user_clk (wb_clk_i),
        .user_rst (wb_rst_i),
        .wb_clk_i (wb_clk_i),
        .wb_rst_i (wb_rst_i),

        // Wishbone (gated)
        .wbs_stb_i (wbs_stb_i_1),
        .wbs_cyc_i (wbs_cyc_i_1),
        .wbs_we_i  (wbs_we_i),
        .wbs_sel_i (wbs_sel_i),
        .wbs_dat_i (wbs_dat_i),
        .wbs_adr_i (wbs_adr_i),
        .wbs_dat_o (wbs_dat_o_1),
        .wbs_ack_o (wbs_ack_o_1),

  // Scan/Test
  .ScanInCC  (io_in[4]),
  .ScanInDL  (io_in[1]),
  .ScanInDR  (io_in[2]),
  .TM        (io_in[5]),
  .ScanOutCC (io_out[0]),

        // Analog / bias pins
        // To give mprj1 its own bias pins, change the indices below.
        .Iref          (analog_io[0]),
        .Vcc_read      (analog_io[1]),
        .Vcomp         (analog_io[2]),
        .Bias_comp2    (analog_io[3]),
        .Vcc_wl_read   (analog_io[12]),
        .Vcc_wl_set    (analog_io[5]),
        .Vbias         (analog_io[6]),
        .Vcc_wl_reset  (analog_io[7]),
        .Vcc_set       (analog_io[8]),
        .Vcc_reset     (analog_io[9]),
        .Vcc_L         (analog_io[10]),
        .Vcc_Body      (analog_io[11])
    );

    // ------------------------------------------------------------
    // Wishbone mux (selected slave drives the return bus)
    // ------------------------------------------------------------
    assign wbs_ack_o = (sel_mprj0 ? wbs_ack_o_0 : 1'b0)
                     | (sel_mprj1 ? wbs_ack_o_1 : 1'b0);

    assign wbs_dat_o = sel_mprj0 ? wbs_dat_o_0 :
                       sel_mprj1 ? wbs_dat_o_1 :
                       32'h0000_0000;

    // Tie off unused outputs to avoid lints (customize as needed)
    assign la_data_out = 128'b0;
    assign io_out      = {`MPRJ_IO_PADS{1'b0}};
    assign io_oeb      = {`MPRJ_IO_PADS{1'b1}};
    assign user_irq    = 3'b000;

endmodule
`default_nettype wire


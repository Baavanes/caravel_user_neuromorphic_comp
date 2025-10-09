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


    // -----------------------------
    // Instantiate your hard macro
    // -----------------------------
   Neuromorphic_X1_wb mprj (
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

  // Wishbone
  .wbs_stb_i (wbs_stb_i),
  .wbs_cyc_i (wbs_cyc_i),
  .wbs_we_i  (wbs_we_i),
  .wbs_sel_i (wbs_sel_i),
  .wbs_dat_i (wbs_dat_i),
  .wbs_adr_i (wbs_adr_i),
  .wbs_dat_o (wbs_dat_o),
  .wbs_ack_o (wbs_ack_o),

  // Scan/Test
  .ScanInCC  (io_in[4]),
  .ScanInDL  (io_in[1]),
  .ScanInDR  (io_in[2]),
  .TM        (io_in[5]),
  .ScanOutCC (io_out[0]),

  // Analog / bias pins (drive from analog_io[] wires you already built)
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


endmodule
`default_nettype wire



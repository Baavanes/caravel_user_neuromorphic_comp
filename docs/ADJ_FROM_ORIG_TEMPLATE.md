# Adjustments from Original Template

Unfortunately, the [original template](https://github.com/chipfoundry/caravel_user_Neuromorphic_X1_32x32) for this [contest](https://chipfoundry.io/challenges/bmlabs) had some bugs. After many, many long hours of debugging, the condensed changes and a post-mortem are below. Note that this note replaced `MACRO_PDN_GRID_FIX.md`, `PDN_BUG_ROOT_CAUSE.md`, and `SPINNER_PDN_FIX.md`, which were all written in the process of debugging.

Many thanks to everyone in the LibreLane and ChipFoundry community, especially @d.m.bailey on the ChipFoundry Matrix chat, for being so patient and helpful!

## TL;DR
 - The custom PDN script `openlane/user_project_wrapper/pdn_override_2.tcl` was replaced with a precisely modified version of the [default script](https://github.com/librelane/librelane/blob/main/librelane/scripts/openroad/common/pdn_cfg.tcl).
 - Various modifications made to `config.json`, including reverting [illegal modifications to PDN strap widths and heights and core ring offset](https://caravel-user-project.readthedocs.io/en/latest/#user-project-wrapper-requirements).
 - On `lvs/lvs_config.json`, added `Neuromorphic_X1_wb` to `EXTRACT_ABSTRACT` and `$PDK_ROOT/$PDK/libs.ref/sky130_fd_sc_hd/spice/sky130_ef_sc_hd__decap_40_12.spice` to `LVS_SPICE_FILES`.
 - **Updated digital IO pins for the Neuromorphic IP** in `verilog/rtl/user_project_wrapper.v` to not overlap with GPIO 0-4, as [otherwise they cannot be preset as output/input](https://caravel-user-project.readthedocs.io/en/latest/#gpio-configuration). Specifically, **incremented the prexisting indices for io_in/io_out by 20**. In other words, now the pin connections are:
   - ScanInCC:  io_in[24]
   - ScanInDL:  io_in[21]
   - ScanInDR:  io_in[22]
   - TM:        io_in[25]
   - ScanOutCC: io_out[20]
 - Tied `io_oeb` in `verilog/rtl/user_project_wrapper.v` to the correct values to allow the Neuromorphic IP's output pin to be actually output data.
 - Updated `verilog/rtl/user_defines.v` to automatically preset the corresponding IO pins on the Neuromorphic IP to be input or output.

See below for the longer post-mortems.

## Custom PDN Script

The template overrode the PDN script. To the best of my knowledge, this was due to the `Neuromorphic_X1_wb` macro needing metal connections from 3 to 4 instead of 4 to 5. However, this custom script forgot to implement many other things the default script takes care of, including supplying power to any and all standard cells.

This works out of the box, however, because the `config.json` was set to use the [Macro-First Hardening Strategy](https://librelane.readthedocs.io/en/latest/usage/caravel/index.html#macro-first-hardening-strategy). This meant that the macro was directly wired to its respective pins without any standard cells. However, as soon as one wanted to have any extra logic at the top level by using the [Top-Level Integration Strategy](https://librelane.readthedocs.io/en/latest/usage/caravel/index.html#top-level-integration-strategy), there were many LVS errors as such logic was not powered. These fails also became apparent when using the template's multi_inst versions of `user_project_wrapper.v` and `config.json`.

Just manually adding the required connections, however, leads to later issues with `make-precheck`. For instance, the custom PDN script did not correctly include all 8 power rails. Additionally, `config.json` had different PDN strap widths and heights along with core offsets than expected, which was being caught by the XOR.

In any case, the solution was isolating the reason for a custom script --- connecting power to a macro on a different layer. Instead of rewriting the script from scratch, the [default PDN script](https://github.com/librelane/librelane/blob/main/librelane/scripts/openroad/common/pdn_cfg.tcl) was copied and a single line was modified to solve this macro power connection issue. This alone solved many of the headaches from physical design.

## LVS Config

The Neuromorphic IP is a black-box macro, and because of this was causing problems in the LVS prechecks. The fix was specifying that `Neuromorphic_X1_wb` should be treated as a black-box by adding it to `EXTRACT_ABSTRACT` in `lvs_config.json`.

Even after, there were still some CVC and LVS errors. These turned out to be because another standard cell was being black-boxed due to lack of a source file: `sky130_ef_sc_hd__decap_40_12`. Adding it to `LVS_SPICE_FILES` in `lvs_config.json` fixed the remaining LVS and CVC errors.

## Digital IO Pins for the Neuromorphic

The `Neuromorphic_X1_wb` has 5 digital ports. In the template, they were mapped to `io_*[4:0]`. The [Caravel harness](https://caravel-user-project.readthedocs.io/en/latest/#gpio-configuration) allows for the initial configuration of IO pins except for `4:0`. "Manual" configuration could be avoided by setting up the initial configuration in `user_defines.v` and moving the pins. In other words, by being on other pins, these ports could automatically be input/output pins upon receiving power, as likely intended.

Moreover, despite having an output digital IO pin, `io_oeb` was not correctly set low to [allow it to output data](https://chipfoundry.io/knowledge-base/connecting-gpios). This was caught by the `OEB` precheck.

Therefore:
 - The digital pins were moved from `4:0` to `24:20`, after the analog pins (which have to be on Caravel's predefined analog IO pins).
 - `io_oeb` was correctly connected to low for specifically `ScanOutCC`.
 - `user_defines.v` was updated to automatically configure the corresponding GPIO pins to their correct state for the `Neuromorphic_X1_wb`.
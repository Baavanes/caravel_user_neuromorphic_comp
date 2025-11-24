# Physical Design Notes

To be complete, and just in case, here are some written-out metrics on this design.

## Timing per PVT Corner
`ss_100C_1v60` has many setup violations, along with max cap and max slew violations.

`tt_025C_1v80` has no setup or hold violations, no max cap violations, and 2 max slew violations.
 - These 2 max slew violations are specifically to analog pins of the Neuromorphic IP, `Vcc_read` and `Vcc_wl_read`. As such, it was deemed okay to ignore them.

`ff_n40C_1v95` has no timing violations.

## Antennas
Max Partial / Required of around 3.5. This is a little more than LibreLane's [recommended max of 3](https://librelane.readthedocs.io/en/latest/usage/caravel/index.html#openroad-checkantennas). However, given that this (or at least this version of the) **design is not meant to be mass produced**, bulk impacts on yield are relatively trivial and it was considered okay to proceed.
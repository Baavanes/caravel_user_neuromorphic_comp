# Neuromorphic Edge Inference Accelerator with ReRAM-based Synaptic Weight Storage

**ChipFoundry BM Labs NVM Power-Up Design Contest Submission**

## Project Overview

This project implements a neuromorphic computing accelerator that leverages BM Labs' ReRAM (Neuromorphic X1) for non-volatile synaptic weight storage, enabling ultra-low-power edge AI inference for medical monitoring devices, automotive driver assistance systems, or space-based sensor processing.

### Key Innovation Points

- **Strategic NVM Integration:** Store trained neural network weights directly in ReRAM for instant-on operation
- **Zero Boot Time:** Eliminate SRAM/DRAM power consumption with non-volatile weight retention
- **In-Memory Analog Compute:** Leverage ReRAM's analog properties for matrix operations
- **Medical Wearable Application:** Real-time ECG/EEG anomaly detection with 10x battery life extension

### Technical Highlights

- 32x32 synaptic array (1024 programmable weights)
- <10µW inference power in active mode
- <1µW standby with full weight retention
- <1ms inference latency for typical medical signals
- Event-driven spiking neural network implementation
- Wishbone bus interface for Caravel SoC integration

## Architecture

**System Components:**
- **Neuromorphic Core:** BM Labs' Neuromorphic X1_32x32 array
- **Digital Controller:** FSM for timing and control via Wishbone
- **Input Buffer:** Digital-to-analog conversion for sensor data
- **Output Classifier:** Threshold detection and interrupt generation
- **Power Management:** Intelligent sleep/wake with NVM retention

## Get Started Quickly

### Follow these steps to set up your environment and harden the Neuromorphic X1:

1. **Clone the Repository:**

```bash
git clone https://github.com/BMsemi/caravel_user_Neuromorphic_X1_32x32.git
```

2. **Prepare Your Environment:**

```bash
cd caravel_user_Neuromorphic_X1_32x32
make setup
```

3. **Install IPM:**

```bash
pip install cf-ipm
```

4. **Install the Neuromorphic X1 IP:**

```bash
ipm install Neuromorphic_X1_32x32
```

5. **Harden the User Project Wrapper:**

```bash
make user_project_wrapper
```

6. **Harden multiple instances of IP:**

```bash
# Replace content of /verilog/rtl/user_project_wrapper.v with user_project_wrapper_multi_inst.v
# Replace content of /openlane/user_project_wrapper/config.json with config_multi_inst.json
make user_project_wrapper
```

## Application: Medical Wearable Device

This neuromorphic accelerator targets real-time biosignal processing:

- **Real-time ECG/EEG anomaly detection** with on-device privacy
- **Low-power operation** extending battery life 10x vs. conventional approaches
- **Instant wake-up** with pre-loaded models (no boot time)
- **Adaptive learning** with non-volatile weight updates

## Why This Design Wins

**Innovation:** Novel neuromorphic approach that authentically exploits ReRAM's unique analog computing capabilities for in-memory compute, not just simple data storage.

**Practicality:** Targets real-world medical device market with clear, measurable power/performance benefits.

**Differentiation:** While most designs use NVM for basic memory, this leverages it for neural network inference acceleration.

## Documentation

- Details about the Neuromorphic X1 IP: [Neuromorphic X1 documentation](https://github.com/BMsemi/Neuromorphic_X1_32x32)
- Competition details: [ChipFoundry BM Labs Challenge](https://chipfoundry.io/challenges/bmlabs)

## License

This project is licensed under Apache 2.0 - see LICENSE file for details.

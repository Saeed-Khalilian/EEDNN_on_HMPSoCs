# Early-Exit DNN Inference on HMPSoCs

This repository contains the source code associated with the following paper:

**"Early-Exit DNN Inference on HMPSoCs"**  

📄 **[Link to the paper]**

## 🔧 Dependencies

This project is built on top of the [ARM-CO-UP](https://github.com/Ehsan-aghapour/ARM-CO-UP) framework, which provides the core infrastructure for efficient DL inference on heterogeneous Multi-Processing SoCs (HMPSoCs).

**As part of this work**, the `earlyexit` branch of ARM-CO-UP was developed to support early-exit DNN architectures tailored to this project.  
This customized version of the framework is included as a Git submodule in the `ARM-CO-UP/` directory.

🔗 For general-purpose use of the ARM-CO-UP framework and additional branches, please refer to the main repository:
https://github.com/Ehsan-aghapour/ARM-CO-UP


---



## Cloning Instructions

To clone this repository along with the correct version of the submodule:

```bash
git clone --recurse-submodules https://github.com/Saeed-Khalilian/EEDNN_on_HMPSoCs.git
```

If you've already cloned the repository:

```bash
git submodule update --init --recursive
```

## 🧪 Workflow: Profiling and Optimization

This project extends the ARM-CO-UP framework by implementing early-exit DNN models and optimizing their deployment across heterogeneous processors using a genetic algorithm (GA).

### 1. Profiling Early-Exit Models

Early-exit models are implemented in the directory:

`ARM-CO-UP/examples/EarlyExit/`

To profile a specific model, use the tuning script:

`ARM-CO-UP/Run_CO-UP`

You can select your desired model from the available options in the `EarlyExit` directory.  
This step profiles the execution time of the model's backbone and early-exit branches on the target hardware.

### 2. Genetic Algorithm for Optimal Mapping

After profiling, the execution time data is passed to the GA algorithm located at:

`python/GA.py`

This algorithm searches for an optimal mapping of the model blocks — including early-exit branches — to available processors (e.g., big/LITTLE cores, GPU).  
The goal is to maximize performance and/or efficiency by exploring different configurations based on the profiling data.

### 3. Running the Optimized Configuration

Once the optimal configuration is found, it can be executed on the device using the same script:

`ARM-CO-UP/Run_CO-UP`

Before running the model, edit the run-time configuration in the Run_CO-UP script to apply your desired setup — for example, the optimized processor mapping found by the GA algorithm.


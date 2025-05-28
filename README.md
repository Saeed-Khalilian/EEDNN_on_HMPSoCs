# Early-Exit DNN Inference on HMPSoCs

This repository contains the source code associated with the following paper:

**"Early-Exit DNN Inference on HMPSoCs"**  

📄 **[Link to the paper]**

## 🔧 Dependencies

This project is built on top of the [ARM-CO-UP](https://github.com/Ehsan-aghapour/ARM-CO-UP) framework, which provides the core infrastructure for efficient DL inference on heterogeneous Multi-Processing SoCs (HMPSoCs).

As part of this work, the `earlyexit` branch of ARM-CO-UP was developed to support early-exit DNN architectures tailored to this project.  
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

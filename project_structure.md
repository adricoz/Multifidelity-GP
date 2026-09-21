# 📁 Multifidelity-GP-Clean - Project Structure

## Quick Overview

## Directory Structure

```
Multifidelity-GP-Clean/  
├──  **.gitignore**
├── charts/
│   └── 🖼️ relations.png
├── example/
│   ├── hartmann_6d/
│   │   ├── convergence_plot.png
│   │   ├── ego_backup.json
│   │   ├── hartmann_6d.py
│   │   ├── Hartmann6d.py
│   │   ├── logfile.log
│   │   └── response_surface_2d.png
│   └── hydrofoil_optim/
│   │   ├── convergence_plot.png
│   │   ├── ego_backup.json
│   │   ├── hydrofoil_optim.py
│   │   ├── logfile.log
│   │   ├── optim_neuralfoil.py
│   │   └── response_surface_2d.png
├── 📂legacy/
│   ├── legacy_nested/
|   |   └── ...
│   └── legacy_non_nested/
│   │   └── ...
├── 📂 mfego/                   \# Project's main source code
│   ├── convergence_plot.png
│   ├── ego_backup.json
│   ├── logfile.log
│   ├── main.py
│   ├── response_1d.png
│   └── 📁 src/
│   │   ├── 📄 acquisition.py
│   │   ├── 📄 data_management.py
│   │   ├── 📄 kernels.py
│   │   ├── 📄 optimizer.py
│   │   ├── 📄 simulator.py
│   │   ├── 📄 surrogate_models.py
│   │   └── 📄 visualization.py
├── 📖 **README.md**
├── 📂 sandbox/
│   └── 📄 gaussian_process_singlefidelity_sandbox.ipynb


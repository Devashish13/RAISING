
# RAISING Repository

This repository contains the source code, simulation data, and documentation for the **RAISING**, a two-stage neural network(NN) implementation framework that first performs hyperparameter tuning to devise the best NN architecture and then performs training on the entire data to estimate the feature importance.

## Repository Structure

```plaintext
- sim1.csv
- docs/
  - RAISING_Demo.html
  - README_hp_optimization.html
  - RAISING_Demo.py
- RAISING/
```

### 1. `sim1.csv`
- **Description**: This file contains the simulated data used for generating tutorial of RAISING.

### 2. `docs/`
- **Description**: The `docs` folder contains all the documentation and demonstration files for the RAISING, including:
  - **`RAISING_Demo.html`**: A tutorial for installing and running RAISING on simulated data.
  - **`README_hp_optimization.html`**: Documentation of the functions describing hyperparameter optimization(STAGE 1) and feature selection(STAGE 2) of RAISING.
  - **`RAISING_Demo.py`**: A Python script similar to **`RAISING_Demo.html`**

### 3. `RAISING/`
- **Description**: The `RAISING` folder contains the source code for the framework. It holds all the necessary scripts and utilities required to run the software.

## Getting Started

### 1. Run the Demo
- Start by exploring the demo in the `docs/` folder:
  - Open **`RAISING_Demo.html`** for a visual walkthrough of the framework.
  - Alternatively, run **`RAISING_Demo.py`** to see how the framework works with the provided simulated data (`sim1.csv`).

### 2. Read the Documentation
- For further details on the RAISING framework, refer to **`README_hp_optimization.html`**.

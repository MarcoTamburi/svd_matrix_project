# CD SVD Matrix Project

## Overview

This project provides a modular Python pipeline for the analysis of Circular Dichroism (CD) data using:

* Singular Value Decomposition (SVD)
* Thermodynamic unfolding models (2, 3, and 4 states)
* Spectral reconstruction and model validation

Unlike traditional pipelines based on raw instrument files, this version is **matrix-driven**:
the user provides a pre-built spectral matrix, and the software handles decomposition, fitting, and reconstruction.

The goal is to offer a **reproducible, user-friendly, and extensible framework** suitable for advanced spectroscopic analysis.

---

## Project Structure

```
svd_matrix_project/
├── src/                # Core computational modules
├── notebooks/          # User interface (step-by-step workflow)
├── configs/            # Configuration files
├── params/             # Editable model parameters
├── data/               # Input data and intermediate files
├── results/            # Output of fitting runs
├── README.md
```

---

## Input Data Format

The software expects a spectral matrix in CSV format:

* First column: `Wavelength`
* Remaining columns: temperatures (numeric)
* Values: CD signal

Example:

```
Wavelength,20,25,30,35
220,-3.1,-2.8,-2.4,-1.9
221,-3.0,-2.7,-2.3,-1.8
```

This file should be placed in:

```
data/user_spectral_matrix.csv
```

---

## Workflow

The pipeline is organized into three notebooks.

### 1. `01_n_components.ipynb`

* Loads and validates the input matrix
* Performs SVD decomposition
* Allows selection of number of components
* Saves:

  * `U_prime.csv`
  * `V_prime.csv`
  * `configs/session_config.json`

This step must be executed when:

* the input matrix changes
* the number of components is redefined

---

### 2. `02_run_fit.ipynb`

* Reads `session_config.json`
* Automatically selects the correct model:

  * 2-state
  * 3-state
  * 4-state
* Runs the thermodynamic fit
* Saves results in:

```
results/fit{n}_run/fit{n}_<timestamp>/
```

Each run includes:

* fitted parameters
* configuration snapshot
* fit diagnostics and plots

---

### 3. `03_spectral_reconstruction.ipynb`

* Loads the latest completed run
* Reconstructs:

  * pure state spectra
  * populations vs temperature
  * full spectra
* Compares experimental vs reconstructed data
* Computes error metrics

This step does **not** rerun the fit.

---

## Models

The following thermodynamic models are implemented:

### 2-state model

```
S1 ⇌ S2
```

Parameters:

* Tm1
* ΔH1

---

### 3-state model (sequential)

```
S1 ⇌ S2 ⇌ S3
```

Parameters:

* Tm1
* dTm12
* ΔH1, ΔH2

---

### 4-state model (sequential)

```
S1 ⇌ S2 ⇌ S3 ⇌ S4
```

Parameters:

* Tm1
* dTm12, dTm23
* ΔH1, ΔH2, ΔH3

---

## Key Features

* Matrix-based input (no dependency on raw instrument files)
* Modular architecture (clear separation between interface and core logic)
* Reproducible runs (each fit stored with full configuration)
* Flexible parameter handling (CSV/XLSX editable)
* Generalized pipeline for multiple thermodynamic models

---

## Design Philosophy

* Notebooks are used as user interface only
* All scientific logic is implemented in `src/`
* Data, configuration, and results are strictly separated

The development follows a clear priority:

1. Software robustness and reproducibility
2. Numerical stability of the fit
3. Physical interpretability of results

---

## Requirements

* Python 3.10+
* numpy
* pandas
* scipy
* scikit-learn
* matplotlib
* openpyxl
* ipywidgets

---

## Future Work

* Validation and tuning of the 4-state model
* Full integration of the 2-state model in reconstruction
* Improved numerical stability and uncertainty estimation
* Extended support for automated analysis workflows

---

## Author

Marco Tamburi
MSc Physics – Data Analysis & Scientific Computing


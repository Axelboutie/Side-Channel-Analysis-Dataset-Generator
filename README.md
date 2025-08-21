# Side-Channel Analysis Dataset Generator

A Rust library for generating and analyzing side-channel attack datasets, with Python bindings. This project provides tools for creating datasets similar to ASCAD (ANSSI-FR) format, implementing various side-channel analysis techniques and countermeasures.

## Features

- Dataset generation with configurable parameters
- Multiple leakage models support:
  - Hamming Weight
  - Weighted Hamming Weight
- Countermeasure implementations:
  - Masking schemes (Boolean, Multiplicative, Affine)
    - Masking option (Fixed)
  - Hiding techniques (shuffling, random delays)
- Analysis tools:
  - Correlation Power Analysis (CPA)
  - Signal-to-Noise Ratio (SNR) analysis
  - Guessing Entropy calculation
- HDF5 file format output compatible with ASCAD format
- Python bindings for easy integration

## Requirements

- Rust 1.88 or higher
- Python 3.12 or higher
- Required Rust dependencies:
  - rayon
  - ndarray
  - hdf5
  - pyo3
  - statrs
  - rand
- Required Python packages:
  - numpy
  - h5py
  - maturin
  - tqdm to use the python file

## Installation

### Rust and Python usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sca_dataset.git
cd sca_dataset
```

2. Build the Rust library with Python bindings directly to use:
```bash
maturin develop -b pyo3 -r
```

3. Build the Rust crate into python packages:
```bash
maturin build -b pyo3 -r
```

### Python usage in your file

1. To use only the library in python
```bash
pip install sca_dataset
```

## Usage

### Warning

If you are using the Rust librairy in developpement mode, with the maturin develop command, the specification of the name of the parameters might not works

### Python Interface

```python
from sca_dataset import Dataset, snr, cpa, ge

# Create a new dataset configuration
dataset = Dataset(
    leakage_model = "Hamming_Weight",   # Leakage model
    hiding = ["shuffle", "delay"],      # Hiding countermeasures
    masking = ["Boolean", "2"],         # Masking scheme and order
    nb_traces = 10000,                  # Number of traces
    nb_samples = 1000,                  # Number of samples
    nb_bytes = 16                       # Number of bytes
)

# Generate dataset and save to HDF5 file
dataset.gen_file(
    snr = 30.0,                    # Signal-to-noise ratio
    weights = None,                # Optional weights for HW
    delay = 10,                    # Random delay range
    filename = "dataset.h5",       # Output filename
    nb_prof=5000,                  # Profiling traces count
    nb_att=5000                    # Attack traces count
    random_rates = 0.30            # If you are using the fixed_pattern set the random rates in the pattern
    testing_set = True             # Set a new group in the dataset for testing
    nb_testing = 1000              # If you enabled the testing set of traces
)
```

### Available Analysis Functions

```python
# Perform SNR analysis
snr_values = snr(labels, traces)

# Perform CPA attack
key, correlations = cpa(traces, plaintexts)

# Calculate guessing entropy
ranks, trace_counts = ge(traces, plaintexts, iterations, key, points)
```

## Example Usage

### Basic Dataset Generation



```python
import gen_dataset as d
import numpy as np
from tools import gen_p

# Configuration
nb_traces_attack = 10000
nb_traces_profiling = 50000
nb_samples = 700
nb_bytes = 16

# Create dataset instance
dataset = d.Dataset(
    leakage_model = "Hamming_Weight",  # Leakage model
    hiding = [],                       # No hiding countermeasures
    maskig = [],                       # No masking
    nb_traces = nb_traces_attack,
    nb_samples = nb_samples,
    nb_bytes = nb_bytes,
    fixed_pattern = True
)

# Generate random plaintexts
plaintext = np.zeros((nb_traces_attack, nb_bytes), dtype=int)
for i in range(nb_traces_attack):
    for j in range(nb_bytes):
        plaintext[i][j] = gen_p()

# Generate traces with batching into a HDF5 file with Python
from batch import batch
batch(
    nbt_a=nb_traces_attack,
    nbt_p=nb_traces_profiling,
    nbs=nb_samples,
    nb_octet=nb_bytes,
    data=dataset,
    batch_size=10000,     # Batch size
    batch=True,           # Enable batching
    weights=[],           # No weighting
    delay=5               # Delay value for delay counter-measure
)

# Generate traces into a HDF5 file with the library

dataset.gen_file(
    snr = 30.0
    weights = []
    delay = 0
    filename = "./result/dataset.h5"
    nb_prof = 50000
    nb_att = 10000
    random_rates = 0.30   # If you are using the fixed_pattern
    testing_set = True
    nb_testing = 1000     # If you enabled the testing set of traces
)
```

### SNR Analysis Example

```python
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sca_dataset as d

# Load generated dataset
with h5py.File("./resultat/test_batch.h5", "r") as f:
    traces = f["Attack_traces/traces"][:]
    labels = f["Attack_traces/labels"][:]
    metadata = f["Attack_traces/metadata"][:]

labels = np.array(labels)

# Calculate SNR Anova on 1 byte
snr_values = d.anova_snr(labels[:,0], traces)  # 256 possible values for byte

# Plot results
plt.plot(snr_values)
plt.title("SNR Anova Analysis")
plt.xlabel("Sample Point")
plt.ylabel("SNR")
plt.show()
```

## Project Structure

- `src/`
  - `lib.rs` - Main library interface and Python bindings
  - `aes_sbox.rs` - AES sbox implementation and utilities
  - `cpa.rs` - Correlation Power Analysis implementation
  - `ge.rs` - Guessing Entropy calculation
  - `anova_snr.rs` - ANOVA Signal-to-Noise Ratio (SNR) analysis
  - `trace.rs` - Trace generation functionality
  - `utils.rs` - Utility functions and helpers
- `python`
  - `gen_dataset`
    - `main.py` - Main file to generate dataset
    - `config.py` - File to config each parameter
    - `gen_file.py` - Generation of HDF5 and utilities for batch
    - `batch.py` - Batching for generation of HDF5
    - `run_security_test.py` - Function to test the counter measure

## Documentation

Generate and view the documentation locally:

```bash
cargo doc --open
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## Acknowledgments

- ANSSI-FR for the ASCAD dataset format
- Nathan ROUSSELOT and Karine HEYDEMANN for the internship and all the knowledge and the explanation of this field

## Author
- Axel BOUTIE 
# PyTorchDedispersion

## Table of Contents
1. [Description](#description)
2. [Features](#features)
3. [Installation](#installation)
   - [Prerequisites](#prerequisites)
   - [Install PyTorch](#install-pytorch)
   - [Install PyTorchDedispersion](#install-pytorchdedispersion)
4. [Usage Instructions](#usage-instructions)
   - [Supported File Formats](#supported-file-formats)
   - [Configuration File Parameters](#configuration-file-parameters)
   - [Running the Dedispersion Script](#running-the-dedispersion-script)
   - [Output](#output)
5. [Running Tests](#running-tests)
6. [License](#license)
7. [Contact Information](#contact-information)

## Description

PyTorchDedispersion is a Python package for processing and analyzing radio telescope data. It provides GPU-accelerated (or CPU) dedispersion, boxcar filtering, and candidate detection, making it suitable for searching for fast radio bursts (FRBs).

## Features

- GPU-accelerated dedispersion using PyTorch (falls back to CPU if CUDA not available)
- Boxcar filtering
- Candidate detection based on SNR thresholds
- Support for skipping bad channels
- Works with OVRO-LWA produced HDF5 and (optionally) Sigproc Filterbank (`.fil`) files
- Configurable via JSON

## Installation

### Prerequisites

If you plan to run on GPU, install **CUDA** and **cuDNN** first. See:
- NVIDIA CUDA Toolkit: https://docs.nvidia.com/cuda/
- NVIDIA cuDNN: https://developer.nvidia.com/cudnn

> CPU-only use does **not** require CUDA/cuDNN.

### Install PyTorch

Pick the build you need from https://pytorch.org/.

Examples:

- **CPU only**
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu


- **CUDA 12.1** 

  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

Verify:

```python
import torch
print("Is CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
```

### Install PyTorchDedispersion

1. **Clone the repository**

   ```bash
   git clone https://github.com/nkosogor/PyTorchDedispersion.git
   cd PyTorchDedispersion
   ```

2. **Install the package**

   * Base install (HDF5 support):

     ```bash
     pip install .
     ```
   * If you also want `.fil` support via the `your` library:

     ```bash
     pip install .[fil]
     ```


## Usage Instructions

### Supported File Formats

* **HDF5** (`.hdf5`)
* **Sigproc Filterbank** (`.fil`) — requires installing the optional extra: `pip install .[fil]`

### Configuration File Parameters

```json
{
  "SOURCE": "/path/to/data/file.hdf5",
  "SNR_THRESHOLD": 7,
  "BOXCAR_WIDTHS": [1, 2, 4, 8, 16],
  "DM_RANGES": [
    {"start": 100, "stop": 200, "step": 0.5},
    {"start": 200, "stop": 500, "step": 1}
  ],
  "BAD_CHANNEL_FILE": "/path/to/bad_channel_file.txt"
}
```

* **SOURCE** *(str)*: Path to input data file (`.hdf5` or `.fil` if the `fil` extra is installed).
* **SNR\_THRESHOLD** *(float)*: Minimum SNR for candidate detection.
* **BOXCAR\_WIDTHS** *(list\[int])*: Widths (in samples) for boxcar filtering.
* **DM\_RANGES** *(list)*: One or more ranges of dispersion measures (pc/cm³).
* **BAD\_CHANNEL\_FILE** *(optional, str)*: Path to a whitespace-separated list of bad channel indices to skip.

> For very large HDF5 files you can use the alternative CLI (`pydedisp-large` or `pydedisp-large2`) and add `LOAD_FREQ_BATCH_SIZE` to the config to process frequency slices efficiently.

### Running the Dedispersion Script

You can use the installed console scripts **or** the module entry point.

* **Console scripts** (recommended):

  ```bash
  # Standard pipeline
  pydedisp -c /path/to/config.json --gpu 0 -v

  # Large HDF5 variants (frequency-sliced), optional:
  pydedisp-large -c /path/to/config.json --gpu 0 -v
  pydedisp-large2 -c /path/to/config.json --gpu 0 -v
  ```

* **Module invocation** (no console scripts):

  ```bash
  python -m pytorch_dedispersion.dedisperse_candidates -c /path/to/config.json --gpu 0 -v
  ```

**Common flags**

* `--gpu <index>`: Choose GPU index (defaults to 0). If CUDA is not available, the code runs on CPU.
* `-v/--verbose`: Print verbose progress.
* `--remove-trend --window-size <N>`: Optional baseline removal before SNR (moving average over `N` samples).

### Output

A CSV file named `candidates_YYYYMMDD-HHMMSS.csv` containing:

```
SNR,Sample Number,Time (sec),Boxcar Width,DM Value
11.5,1601,2.03,1,476
```

* **SNR** — Signal-to-noise ratio
* **Sample Number** — Index of the time sample
* **Time (sec)** — Sample time in seconds
* **Boxcar Width** — Boxcar width (samples)
* **DM Value** — Dispersion measure (pc/cm³)

## Running Tests

We ship both unit and integration tests. To run the full test suite:

```bash
# from repo root
pip install -e .[dev,fil]
pytest
```

If you prefer to run the existing scripts directly:

```bash
cd tests
python integration_test.py
python dedispersion_test.py
```

> The integration test runs the CLI and checks that a `candidates_*.csv` file is produced with expected values. Tests will use GPU if available; otherwise they run on CPU.

## License

BSD 3-Clause License. See [LICENSE](LICENSE).

## Contact Information

Send questions to **[nakosogorov@gmail.com](mailto:nakosogorov@gmail.com)**.




# PyTorchDedispersion

## Description

PyTorchDedispersion is a Python package designed for processing and analyzing radio telescope data using GPU acceleration. It provides tools for dedispersion, boxcar filtering, and candidate detection, making it suitable for searching for fast radio bursts (FRBs).

## Features

- GPU-accelerated dedispersion using PyTorch
- Boxcar filtering 
- Candidate detection based on SNR thresholds
- Support for handling bad channels
- Configurable through JSON files

## Installation

### Prerequisites
Ensure that CUDA and cuDNN are installed on your system. For detailed installation instructions, refer to the [NVIDIA CUDA Toolkit Installation Guide](https://docs.nvidia.com/cuda/) and the [NVIDIA cuDNN Installation Guide](https://developer.nvidia.com/cudnn).

### Install PyTorch
Find a suitable version of PyTorch at the [PyTorch website](https://pytorch.org/).

Example command for installing PyTorch with CUDA 12.1 support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify the installation:
```python
import torch

print("Is CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
```

Expected output:
```plaintext
Is CUDA available: True
CUDA version: 12.1
cuDNN version: 8902
```

### Install PyTorchDedispersion

1. **Clone the repository:**
    ```bash
    git clone https://github.com/nkosogor/PyTorchDedispersion.git
    cd PyTorchDedispersion
    ```

2. **Install the package and dependencies:**
    ```bash
    pip install .
    ```

## Usage Instructions

### Configuration File Parameters

```json
{
    "SOURCE": ["/path/to/local/file.fil"],
    "SNR_THRESHOLD": 7,
    "BOXCAR_WIDTHS": [1, 2, 4, 8, 16],
    "DM_RANGES": [
        {"start": 100, "stop": 200, "step": 0.5},
        {"start": 200, "stop": 500, "step": 1}
    ],
    "BAD_CHANNEL_FILE": "/path/to/bad_channel_file.txt"
}
```

- **SOURCE**: Input data file path.
- **SNR_THRESHOLD**: Minimum SNR for candidate detection.
- **BOXCAR_WIDTHS**: List of widths (in samples) for boxcar filtering.
- **DM_RANGES**: Ranges of dispersion measures (in pc/cm^3) to search.
- **BAD_CHANNEL_FILE** (Optional): Path to a file with bad channel indices.


### Running the Dedispersion Script

Use the `dedisperse_candidates.py` script to process your data based on the configuration file. Run the script as follows:

```bash
python pytorch_dedispersion/dedisperse_candidates.py --config /path/to/config.json --verbose --gpu 0
```

Additional command-line options:
- `--remove-trend`: Remove trend from the data.
- `--window-size`: Specify window size for trend removal.

### Output

The script generates a CSV file containing candidate information, saved in the format `candidates_YYYYMMDD-HHMMSS.csv`.

## Running Tests

For detailed instructions on running tests, refer to the [tests README](tests/README.md).

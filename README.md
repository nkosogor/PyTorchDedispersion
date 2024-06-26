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

### Supported File Formats
This library supports [Sigproc Filterbank](https://sigproc.sourceforge.net/) format.

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
- **DM_RANGES**: Ranges of dispersion measures (in pc/cm³) to search.
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

The script generates a CSV file containing candidate information, saved in the format `candidates_YYYYMMDD-HHMMSS.csv`. The output CSV file will have lines formatted as follows:

```plaintext
SNR,Sample Number,Time (sec),Boxcar Width,DM Value
11.5,1601,2.03,1,476
```

- **SNR**: The signal-to-noise ratio of the detected candidate.
- **Sample Number**: The sample number where the candidate was detected.
- **Time (sec)**: The corresponding time in seconds for the detected sample.
- **Boxcar Width**: The width of the boxcar (in samples) filter used to detect the candidate.
- **DM Value**: The dispersion measure value in pc/cm³ for the detected candidate.

## Running Tests

For detailed instructions on running tests, refer to the [tests README](tests/README.md).

## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.

## Contact Information

For questions, please contact nakosogorov@gmail.com.


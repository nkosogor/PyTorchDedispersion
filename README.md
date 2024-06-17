# PyTorchDedispersion

## Features
To be added

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

## Running Tests

For detailed instructions on running tests, refer to the [tests README](tests/README.md).



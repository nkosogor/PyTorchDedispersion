# Install

## Prerequisites
Install PyTorch per https://pytorch.org/ (CUDA optional).

## Install the package
    pip install .          # base (.hdf5)
    pip install .[fil]     # add .fil support via `your`

## Dev + tests
    pip install -e .[dev,fil]
    pytest -q


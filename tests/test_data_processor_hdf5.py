import numpy as np
import h5py
import torch
from pathlib import Path
from pytorch_dedispersion.data_processor import DataProcessor

import pytest
from pytorch_dedispersion import file_handler

def _make_tiny_h5(path: Path, ntime=8, freqs_mhz=(85.0, 84.0, 83.0), tInt=0.01):
    # Store MHz in Hz in the file (matches your loader)
    freqs_hz = np.asarray(freqs_mhz, dtype=np.float64) * 1e6
    # I dataset is (nsamples, nchans) in file
    I = np.arange(ntime * len(freqs_mhz), dtype=np.float32).reshape(ntime, len(freqs_mhz))
    with h5py.File(path, "w") as hf:
        obs = hf.create_group("Observation1")
        obs.attrs["tInt"] = tInt
        tuning = obs.create_group("Tuning1")
        tuning.create_dataset("freq", data=freqs_hz)
        tuning.create_dataset("I", data=I)

def test_hdf5_basic(tmp_path):
    f = tmp_path / "tiny.hdf5"
    _make_tiny_h5(f, ntime=5, freqs_mhz=(85.0, 84.0, 83.0), tInt=0.02)

    dp = DataProcessor(str(f))
    dp.load_data()
    # data expected shape: (nchans, nsamples)
    assert dp.data.shape == (3, 5)
    # loader converts to MHz and ensures descending order
    freqs = dp.get_frequencies()
    assert np.allclose(freqs, np.array([85.0, 84.0, 83.0], dtype=np.float64))
    # header basics
    assert dp.header.nchans == 3
    assert dp.header.tsamp == 0.02
    assert dp.header.fch1 == 85.0
    assert np.isclose(dp.header.foff, -1.0)  # descending

def test_hdf5_freq_slice_and_bad_channels(tmp_path):
    f = tmp_path / "tiny2.hdf5"
    _make_tiny_h5(f, ntime=4, freqs_mhz=(90.0, 89.0, 88.0, 87.0), tInt=0.01)

    # Slice the middle two channels [1:3] -> freqs 89,88; then mark one bad
    dp = DataProcessor(str(f), freq_slice=(1, 3), bad_channels=[1])  # bad index is relative to original
    dp.load_data()
    # After masking, only one "good" channel remains
    assert dp.data.shape[0] == 1
    assert dp.header.nchans == 1
    freqs = dp.get_frequencies()
    assert freqs.shape == (1,)
    # still descending (trivial with 1)
    assert freqs[0] in (89.0, 88.0)

def test_hdf5_all_bad_channels(tmp_path):
    f = tmp_path / "tiny3.hdf5"
    _make_tiny_h5(f, ntime=3, freqs_mhz=(81.0, 80.0), tInt=0.005)

    # Mask both channels out
    dp = DataProcessor(str(f), bad_channels=[0, 1])
    dp.load_data()

    assert dp.data.shape == (0, 0)
    assert dp.header.nchans == 0
    assert dp.header.tsamp == 0.005
    assert dp.get_frequencies().size == 0

def test_file_handler_invalid_path(tmp_path):
    bad_path = tmp_path / "does_not_exist.txt"
    fh = file_handler.FileHandler(str(bad_path))
    with pytest.raises(ValueError):
        fh.load_file()

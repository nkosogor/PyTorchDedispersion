# Configuration

    {
      "SOURCE": "/path/to/local/file.fil",
      "SNR_THRESHOLD": 7,
      "BOXCAR_WIDTHS": [1, 2, 4, 8, 16],
      "DM_RANGES": [
        {"start": 100, "stop": 200, "step": 0.5},
        {"start": 200, "stop": 500, "step": 1}
      ],
      "BAD_CHANNEL_FILE": "/path/to/bad_channel_file.txt"
    }

- **SOURCE**: input data (.fil or .hdf5)
- **SNR_THRESHOLD**: minimum SNR for candidate detection
- **BOXCAR_WIDTHS**: boxcar widths (samples)
- **DM_RANGES**: DM search ranges (pc/cm³)
- **BAD_CHANNEL_FILE** *(optional)*: list of bad channel indices to skip

*Large HDF5-only options (for `pydedisp-large` / `pydedisp-large2`):*
- **LOAD_FREQ_BATCH_SIZE**: number of freq channels per batch
- **BOXCAR_BATCH_SIZE**: batch boxcar widths to limit memory


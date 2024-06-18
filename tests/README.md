# Tests for PyTorchDedispersion

This directory contains tests for the PyTorchDedispersion project. These tests ensure that different components of the software work together correctly and that individual functionalities are verified.

## Files

- `test.fil`: A sample `.fil` file used for testing purposes.
- `test_config.json`: A custom configuration file for the integration test.
- `bad_channels.txt`: A file containing indices of channels to be excluded during the dedispersion process.
- `integration_test.py`: The current integration test script.
- `dedispersion_test.py`: The unit test script for dedispersion functionalities.

## Running the Tests

### Steps to Run the Tests

1. **Navigate to the `tests` directory**:
   ```bash
   cd tests
   ```

2. **Run the integration test script**:
   ```bash
   python integration_test.py
   ```

2. **Run the unit tests**:
   ```bash
   python dedispersion_test.py
   ```

### What the Integration Test Does

1. The script runs the `dedisperse_candidates.py` script using the configuration specified in `test_config.json`.
2. It captures and prints the output of the dedispersion process.
3. It verifies that a candidates file (`candidates_*.csv`) is created.
4. It checks that there is a line in the candidates file with the expected values.
5. It removes the created candidates file after the test is completed.

### Expected Output

When you run the test, you should see output similar to the following:

```plaintext
File loaded from: test.fil
Data prepared in 0.02 seconds
Loaded bad channels: [10, 20, 30, 40, 50]
Using device: cuda:0
Initial GPU memory usage:
Memory Allocated: 6.47 MB
Memory Reserved: 22.00 MB
...
Dedispersion and summing completed in 0.31 seconds
GPU memory usage after dedispersion:
Memory Allocated: 3248.66 MB
Memory Reserved: 6488.00 MB
...
Boxcar filtering completed in 0.09 seconds
Candidate finding completed in 0.03 seconds
Candidates saved to candidates_YYYYMMDD-HHMMSS.csv

Test passed. Candidates file created: /path/to/tests/candidates_YYYYMMDD-HHMMSS.csv
Test passed. Expected line found: SNR=13.897394180297852, Time=2.0288829375
All tests passed. Candidates file removed.
```

This output indicates that the dedispersion process completed successfully, the expected candidates file was created, the expected line was found, and the test passed.

### What the Unit Tests Do
The `dedispersion_test.py` script contains several test cases that validate the functionality of the dedispersion process. These tests include:

1. `test_perform_dedispersion`: This test verifies that the `perform_dedispersion` method of the `Dedispersion` class correctly dedisperses the input data and returns a tensor of the expected shape.
2. `test_best_dm`: This test checks that the `best_dm` method of the `Dedispersion` class correctly identifies the dispersion measure that maximizes the signal-to-noise ratio.

### Expected Output

When you run the tests, you should see output similar to the following:
```
Ran 2 tests in 5.598s
OK
```

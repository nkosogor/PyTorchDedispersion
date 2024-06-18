import unittest
import torch
import numpy as np
from pytorch_dedispersion.dedispersion import Dedispersion

def generate_synthetic_data():
    # Parameters
    freq_start = 55  # MHz
    freq_end = 85    # MHz
    channel_width = 0.024  # MHz
    time_resolution = 0.01  # seconds
    max_time = 100  # seconds
    true_dm = 40.0  # pc/cm^3

    # Generate frequency and time arrays
    frequencies = np.arange(freq_start, freq_end, channel_width)
    time_samples = np.arange(0, max_time, time_resolution)

    # Generate synthetic data with noise
    np.random.seed(42)
    data = np.random.normal(0, 0.1, (len(frequencies), len(time_samples)))

    # Constants
    k_dm = 4.148808e3  # MHz^2 cm^3 pc^-1 s

    # Generate strong artificial signal with highest frequency first
    signal_start_time = 50  # seconds
    for i, freq in enumerate(frequencies[::-1]):  # Reverse the order of frequencies
        delay = k_dm * true_dm * (-freq_start**-2 + freq**-2)
        start_sample = int((signal_start_time + delay) / time_resolution)
        if start_sample < len(time_samples):
            data[len(frequencies) - 1 - i, start_sample] = 1.0  # Strong signal

    return data, frequencies, time_samples

class TestDedispersion(unittest.TestCase):
    def test_perform_dedispersion(self):
        data, frequencies, time_samples = generate_synthetic_data()

        # Convert to PyTorch tensors
        data_tensor = torch.from_numpy(data)
        frequencies_tensor = torch.from_numpy(frequencies)
        dm_range = torch.arange(0, 100, 0.5)  # pc/cm^3
        freq_start = frequencies[0]
        time_resolution = time_samples[1] - time_samples[0]

        # Create Dedispersion instance and perform dedispersion
        dedispersion = Dedispersion(data_tensor, frequencies_tensor, dm_range, freq_start, time_resolution)
        dedispersed_data = dedispersion.perform_dedispersion()

        # Check the shape and type of the output
        self.assertIsInstance(dedispersed_data, torch.Tensor)
        self.assertEqual(dedispersed_data.shape, (len(dm_range), len(frequencies), len(time_samples)))

    def test_best_dm(self):
        data, frequencies, time_samples = generate_synthetic_data()

        # Convert to PyTorch tensors
        data_tensor = torch.from_numpy(data)
        frequencies_tensor = torch.from_numpy(frequencies)
        dm_range = torch.arange(0, 100, 0.5)  # pc/cm^3
        freq_start = frequencies[0]
        time_resolution = time_samples[1] - time_samples[0]

        # Create Dedispersion instance and perform dedispersion
        dedispersion = Dedispersion(data_tensor, frequencies_tensor, dm_range, freq_start, time_resolution)
        dedispersed_data = dedispersion.perform_dedispersion()

        # Sum along the frequency axis and find max value along the time axis for each DM
        summed_data = dedispersed_data.sum(dim=1)
        max_values, _ = torch.max(summed_data, dim=1)  # Unpack the tuple

        # Find the best DM
        best_dm_idx = torch.argmax(max_values).item()
        best_dm = dm_range[best_dm_idx].item()

        # Check that the best DM is close to the true DM
        true_dm = 40.0  # pc/cm^3
        self.assertAlmostEqual(best_dm, true_dm, delta=0.1)  # Allow a small error


if __name__ == '__main__':
    unittest.main()
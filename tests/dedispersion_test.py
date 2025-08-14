import os
import unittest
import torch
import numpy as np
from pytorch_dedispersion.dedispersion import Dedispersion

def generate_synthetic_data():
    # Small mode in CI for speed
    small = os.getenv("CI") == "1"

    # Band and sampling
    f_lo, f_hi = 55.0, 85.0           # MHz
    df = 0.048 if small else 0.024    # coarser grid in CI
    dt = 0.02 if small else 0.01      # larger tsamp in CI
    t_max = 30.0 if small else 100.0  # shorter span in CI

    true_dm = 40.0
    k_dm = 4.148808e3  # MHz^2 cm^3 pc^-1 s

    freqs_asc = np.arange(f_lo, f_hi, df)
    freqs = freqs_asc[::-1].copy().astype(np.float32)   # DESCENDING
    t = np.arange(0.0, t_max, dt).astype(np.float32)

    rng = np.random.default_rng(42)
    data = rng.normal(0.0, 0.1, (len(freqs), len(t))).astype(np.float32)

    # Inject dispersed pulse referenced to TOP of band
    f_ref = float(freqs[0])
    t0 = 5.0 if small else 50.0
    for ch, f in enumerate(freqs):
        delay = k_dm * true_dm * (f**-2 - f_ref**-2)
        idx = int((t0 + delay) / dt)
        if idx < len(t):
            data[ch, idx] += 5.0

    return data, freqs, t

class TestDedispersion(unittest.TestCase):
    def test_perform_dedispersion(self):
        data, frequencies, time_samples = generate_synthetic_data()
        data_tensor = torch.from_numpy(data)
        frequencies_tensor = torch.from_numpy(frequencies)
        # coarser DM grid in CI
        dm_step = 1.0 if os.getenv("CI") == "1" else 0.5
        dm_range = torch.arange(0.0, 100.0 + 1e-6, dm_step, dtype=torch.float32)
        freq_start = float(frequencies[0])
        time_resolution = float(time_samples[1] - time_samples[0])

        dedispersion = Dedispersion(data_tensor, frequencies_tensor, dm_range, freq_start, time_resolution)
        out = dedispersion.perform_dedispersion()
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (len(dm_range), len(frequencies), len(time_samples)))

    def test_best_dm(self):
        data, frequencies, time_samples = generate_synthetic_data()
        data_tensor = torch.from_numpy(data)
        frequencies_tensor = torch.from_numpy(frequencies)
        dm_step = 1.0 if os.getenv("CI") == "1" else 0.5
        dm_range = torch.arange(0.0, 80.0 + 1e-6, dm_step, dtype=torch.float32)
        freq_start = float(frequencies[0])
        time_resolution = float(time_samples[1] - time_samples[0])

        dedispersion = Dedispersion(data_tensor, frequencies_tensor, dm_range, freq_start, time_resolution)
        dedispersed_data = dedispersion.perform_dedispersion()

        summed = dedispersed_data.sum(dim=1)     # (n_dm, n_time)
        max_vals = torch.amax(summed, dim=1)
        i = int(torch.argmax(max_vals))
        best_dm = float(dm_range[i])

        # Parabolic refinement (optional but nice)
        if 0 < i < len(dm_range) - 1:
            y0, y1, y2 = max_vals[i-1].item(), max_vals[i].item(), max_vals[i+1].item()
            denom = (y0 - 2*y1 + y2)
            delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
            best_dm = float(dm_range[i]) + delta * float(dm_step)

        # Assert to DM grid resolution
        self.assertLessEqual(abs(best_dm - 40.0), float(dm_step) + 1e-6)

if __name__ == "__main__":
    unittest.main()

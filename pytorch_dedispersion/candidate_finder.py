import torch
import torch.nn.functional as F

class CandidateFinder:
    def __init__(self, boxcar_data, window_size=50):
        self.boxcar_data = boxcar_data
        self.window_size = window_size

    def find_candidates(self, snr_threshold, boxcar_widths, remove_trend=False):
        candidates = []
        for i, data in enumerate(self.boxcar_data):
            if remove_trend:
                baseline = self.calculate_baseline(data)
                detrended_data = data - baseline[:, :data.shape[1]]
                snr = self.calculate_snr(detrended_data)
            else:
                snr = self.calculate_snr(data)
            above_threshold = snr > snr_threshold
            if above_threshold.any():
                candidate_indices = torch.nonzero(above_threshold, as_tuple=False)
                for idx in candidate_indices:
                    dm_index = idx[0].item()
                    time_sample = idx[1].item()
                    candidates.append({
                        'Boxcar Width': boxcar_widths[i],
                        'DM Index': dm_index,
                        'Time Sample': time_sample,
                        'SNR': snr[dm_index, time_sample].item()
                    })
        return candidates

    def calculate_baseline(self, data):
        """Calculate the baseline using a moving average."""
        pad = self.window_size // 2
        padded_data = F.pad(data, (pad, pad), mode='reflect')
        baseline = F.avg_pool1d(padded_data.unsqueeze(0), kernel_size=self.window_size, stride=1, padding=0).squeeze(0)
        return baseline

    def calculate_snr(self, data):
        """Calculate the signal-to-noise ratio (SNR)."""
        mean = data.mean(dim=1, keepdim=True)
        std = data.std(dim=1, keepdim=True)
        snr = (data - mean) / std
        return snr

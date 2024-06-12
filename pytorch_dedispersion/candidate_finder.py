import torch

class CandidateFinder:
    def __init__(self, boxcar_data):
        self.boxcar_data = boxcar_data

    def find_candidates(self, snr_threshold=10):
        candidates = []
        for i, data in enumerate(self.boxcar_data):
            mean = data.mean(dim=1, keepdim=True)
            std = data.std(dim=1, keepdim=True)
            snr = (data - mean) / std
            
            for dm_idx, dm_snr in enumerate(snr):
                peaks = torch.nonzero(dm_snr > snr_threshold, as_tuple=False)
                for peak in peaks:
                    candidates.append({
                        'Boxcar Width': widths[i],
                        'DM Index': dm_idx,
                        'Time Sample': peak.item(),
                        'SNR': dm_snr[peak].item()
                    })
        return candidates

import torch

class Dedispersion:
    def __init__(self, data_tensor, frequencies_tensor, dm_range, freq_start, time_resolution):
        self.data_tensor = data_tensor
        self.frequencies_tensor = frequencies_tensor
        self.dm_range = dm_range
        self.freq_start = freq_start
        self.time_resolution = time_resolution

    def perform_dedispersion(self):
        k_dm = 4.148808e3  # MHz^2 cm^3 pc^-1 s
        delays = k_dm * self.dm_range[:, None] * (-self.frequencies_tensor**-2 + self.freq_start**-2)
        delay_samples = torch.round(delays / self.time_resolution).long()

        expanded_data = self.data_tensor.unsqueeze(0).expand(len(self.dm_range), -1, -1)
        expanded_delays = delay_samples.unsqueeze(2).expand(-1, -1, self.data_tensor.shape[1])

        time_indices = torch.arange(self.data_tensor.shape[1], device=self.data_tensor.device).unsqueeze(0).unsqueeze(0).expand(len(self.dm_range), self.data_tensor.shape[0], -1)
        shifted_indices = (time_indices - expanded_delays) % self.data_tensor.shape[1]

        return expanded_data.gather(2, shifted_indices)

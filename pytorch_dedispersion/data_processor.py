import numpy as np
import h5py
import your

class DataProcessor:
    def __init__(self, file_path, freq_slice=None, bad_channels=None):
        """
        Initialize DataProcessor
        
        Args:
            file_path (str): Path to the data file.
            freq_slice (tuple, optional): (start, end) indices for frequency channels to load
                                          (Python-style, end is non-inclusive)
            bad_channels (list[int], optional): List of channels to skip. If None, none are skipped
        """
        self.file_path = file_path
        self.header = None
        self.data = None
        self.freq_slice = freq_slice
        self.bad_channels = bad_channels if bad_channels else []

    def load_data(self):
        """
        Load the file and extract header and data
        For .hdf5 files, uses h5py and converts frequencies from Hz to MHz,
        reversing the order if necessary to match typical .fil conventions
        For other file types, uses the existing loader
        """
        if self.file_path.endswith('.hdf5'):
            self._load_hdf5()
        else:
            self._load_fil()

    def _load_fil(self):
        """
        Load a .fil file using the `your` library
        """
        your_object = your.Your(self.file_path)
        self.header = your_object.your_header
        # (nspectra, nchans) -> transpose -> (nchans, nspectra)
        self.data = your_object.get_data(nstart=0, nsamp=self.header.nspectra).T

        # If there are bad channels, remove them from `self.data`.
        if self.bad_channels:
            all_channels = np.arange(self.data.shape[0])
            good_channels = np.array(
                list(set(all_channels) - set(self.bad_channels))
            )
            good_channels.sort()  # keep it sorted
            self.data = self.data[good_channels, :]

        # Build a frequency array based on .fil header
        self.frequencies = self._build_frequency_array()

    def _load_hdf5(self):
        """
        Load data from an HDF5 file. Skips bad channels entirely so they never go to memory.
        Also respects freq_slice if given
        """
        with h5py.File(self.file_path, 'r') as hdf:
            obs = hdf['Observation1']
            tuning = obs['Tuning1']

            # 1. Read the entire freq array (small overhead) so we know which channels exist.
            freq_hz = tuning['freq'][:]
            n_total_chans = freq_hz.shape[0]  # total # of frequency channels in file

            # 2. Figure out the baseline channel indices
            all_channels = np.arange(n_total_chans)

            # 2a. If freq_slice was provided, restrict `all_channels` accordingly.
            #     freq_slice=(start, end) uses Python slicing rules, i.e. [start, end).
            if self.freq_slice is not None:
                start, end = self.freq_slice
                # Clamp end if it's bigger than the total number of channels
                end = min(end, n_total_chans)
                all_channels = all_channels[start:end]

            # 2b. Now remove the bad_channels from that set
            if self.bad_channels:
                bad_channels_set = set(self.bad_channels)
                good_channels = [ch for ch in all_channels if ch not in bad_channels_set]
            else:
                good_channels = all_channels

            # If no good channels remain, set empty data & header, then return
            if len(good_channels) == 0:
                self.data = np.zeros((0, 0), dtype=np.float32)
                self.frequencies = np.zeros((0,), dtype=np.float32)
                # Minimal "header" so code doesn't crash
                class Header:
                    pass
                header = Header()
                header.fch1 = 0.0
                header.foff = 0.0
                header.nchans = 0
                header.tsamp = obs.attrs.get('tInt', 0.001337)
                header.nspectra = 0
                self.header = header
                return

            # 3. Advanced indexing: read only the "good" channels into memory
            freq_hz = freq_hz[good_channels]  # shape: (good_chans,)
            intensity = tuning['I'][:, good_channels]  # shape: (nsamples, good_chans)

            # 4. Convert freq to MHz
            freq_mhz = freq_hz / 1e6

            # 5. Reverse order if needed (many .fil readers expect highest freq first).
            #    Typically you'd check if freq_mhz is ascending, then reverse if ascending:
            if len(freq_mhz) > 1 and freq_mhz[0] < freq_mhz[-1]:
                freq_mhz = np.ascontiguousarray(freq_mhz[::-1])
                intensity = np.ascontiguousarray(intensity[:, ::-1])

            # Build a minimal header object
            class Header:
                pass

            header = Header()
            header.fch1 = freq_mhz[0]
            if len(freq_mhz) > 1:
                header.foff = freq_mhz[1] - freq_mhz[0]
            else:
                header.foff = 0.0
            header.nchans = len(freq_mhz)
            header.tsamp = obs.attrs.get('tInt', 0.001337)
            header.nspectra = intensity.shape[0]

            self.header = header
            # After reversing, the shape is still (nsamples, good_chans).
            # We want shape (nchans, nsamples) to match the typical .fil convention:
            self.data = intensity.T
            self.frequencies = freq_mhz

    def get_frequencies(self):
        """
        Return the frequency array. If loaded, just return `self.frequencies`
        """
        return self.frequencies

    def _build_frequency_array(self):
        """
        Helper to build frequency array for .fil files using fch1, foff, and nchans
        """
        freqs = []
        start_freq = self.header.fch1
        for i in range(self.header.nchans):
            freqs.append(start_freq + i * self.header.foff)
        return np.array(freqs)

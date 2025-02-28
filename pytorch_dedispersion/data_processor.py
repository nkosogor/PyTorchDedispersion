import numpy as np
import h5py
import your  

class DataProcessor:
    def __init__(self, file_path, freq_slice=None):
        """
        Initialize DataProcessor.
        
        Args:
            file_path (str): Path to the data file.
            freq_slice (tuple, optional): (start, end) indices for frequency channels to load.
                                          If None, loads all frequencies.
        """
        self.file_path = file_path
        self.header = None
        self.data = None
        self.freq_slice = freq_slice

    def load_data(self):
        """
        Load the file and extract header and data.
        For .hdf5 files, uses h5py and converts frequencies from Hz to MHz,
        reversing the order if necessary to match fil conventions.
        For other file types, uses the existing loader.
        """
        if self.file_path.endswith('.hdf5'):
            self._load_hdf5()
        else:
            self._load_fil()

    def _load_fil(self):
        your_object = your.Your(self.file_path)
        self.header = your_object.your_header
        self.data = your_object.get_data(nstart=0, nsamp=self.header.nspectra).T

    def _load_hdf5(self):
        with h5py.File(self.file_path, 'r') as hdf:
            obs = hdf['Observation1']
            tuning = obs['Tuning1']
            # If a frequency slice is provided, load only that subset; otherwise load all.
            if self.freq_slice is None:
                intensity = tuning['I'][:]
                freq_hz = tuning['freq'][:]
            else:
                start, end = self.freq_slice
                # Slice along the frequency axis (remember: intensity shape is (nsamples, nchans))
                intensity = tuning['I'][:, start:end]  
                freq_hz = tuning['freq'][start:end]
            freq_mhz = freq_hz / 1e6

            # Standardize frequency order: Fil files expect highest frequency first.
            if freq_mhz[0] < freq_mhz[-1]:
                freq_mhz = freq_mhz[::-1]
                intensity = np.ascontiguousarray(intensity[:, ::-1])


            # Create a simple header object.
            class Header:
                pass
            header = Header()
            header.fch1 = freq_mhz[0]
            header.foff = (freq_mhz[1] - freq_mhz[0]) if len(freq_mhz) > 1 else 0.0
            header.nchans = len(freq_mhz)
            header.tsamp = obs.attrs.get('tInt', 0.001337)
            header.nspectra = intensity.shape[0]
            self.header = header
            self.data = intensity.T  # Now shape (nchans, nsamples)

    def get_frequencies(self):
        fch1 = self.header.fch1
        foff = self.header.foff
        nchans = self.header.nchans
        return np.array([fch1 + foff * i for i in range(nchans)])

# Example usage remains similar:
if __name__ == "__main__":
    file_path = "extracted_beam_290.44_21.88.hdf5"
    # For testing full load, leave freq_slice as None.
    processor = DataProcessor(file_path)
    processor.load_data()
    print("Header Info:")
    print("fch1:", processor.header.fch1)
    print("foff:", processor.header.foff)
    print("nchans:", processor.header.nchans)
    print("tsamp:", processor.header.tsamp)
    print("nspectra:", processor.header.nspectra)
    freqs = processor.get_frequencies()
    print("Frequencies (MHz):", freqs)

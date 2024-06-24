import numpy as np
import your 

class DataProcessor:
    def __init__(self, file_path):
        """
        Initialize DataProcessor.

        Args:
            file_path (str): Path to the data file.
        """
        self.file_path = file_path
        self.header = None
        self.data = None

    def load_data(self):
        """
        Load the file using your package and extract header and data.
        """
        your_object = your.Your(self.file_path)
        self.header = your_object.your_header
        self.data = your_object.get_data(nstart=0, nsamp=self.header.nspectra).T

    def get_frequencies(self):
        """
        Get the frequencies from the header.

        Returns:
            np.ndarray: Array of frequencies.
        """
        fch1 = self.header.fch1
        foff = self.header.foff
        nchans = self.header.nchans
        return np.array([fch1 + foff * i for i in range(nchans)])

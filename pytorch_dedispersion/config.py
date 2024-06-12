import tempfile

# Example settings
SOURCE = "https://zenodo.org/record/3905426/files/FRB180417.fil"  # or "/path/to/local/file.fil"
TEMP_DIR = tempfile.TemporaryDirectory()
DOWNLOAD_DIR = TEMP_DIR.name
SNR_THRESHOLD = 7
BOXCAR_WIDTHS = [1, 2, 4, 8, 16]
DM_RANGE = (300, 800)

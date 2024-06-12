import torch
from pytorch_dedispersion.file_handler import FileHandler
from pytorch_dedispersion.data_processor import DataProcessor
from pytorch_dedispersion.dedispersion import Dedispersion
from pytorch_dedispersion.boxcar_filter import BoxcarFilter
from pytorch_dedispersion.candidate_finder import CandidateFinder
from pytorch_dedispersion.config import SOURCE, DOWNLOAD_DIR, SNR_THRESHOLD, BOXCAR_WIDTHS, DM_RANGE

def main():
    # Load file
    handler = FileHandler(SOURCE, DOWNLOAD_DIR)
    file_path = handler.load_file()

    # Process data
    processor = DataProcessor(file_path)
    processor.load_data()
    data = processor.data
    frequencies = processor.get_frequencies()

    # Dedispersion parameters
    time_resolution = processor.header.tsamp
    freq_start = frequencies[0]
    dm_range = torch.linspace(DM_RANGE[0], DM_RANGE[1], 2 * 400 + 1)

    # Convert data to tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    frequencies_tensor = torch.tensor(frequencies, dtype=torch.float32).to(device)

    # Dedisperse data
    dedisperse = Dedispersion(data_tensor, frequencies_tensor, dm_range, freq_start, time_resolution)
    dedispersed_data = dedisperse.perform_dedispersion()
    summed_data = dedispersed_data.sum(dim=1)

    # Boxcar filtering
    boxcar = BoxcarFilter(summed_data)
    boxcar_data = boxcar.apply_boxcar(BOXCAR_WIDTHS)

    # Find candidates
    finder = CandidateFinder(boxcar_data)
    candidates = finder.find_candidates(SNR_THRESHOLD)

    # Output candidates
    for candidate in candidates:
        print(f"Candidate - Boxcar Width: {candidate['Boxcar Width']}, DM Index: {candidate['DM Index']}, "
              f"Time Sample: {candidate['Time Sample']}, SNR: {candidate['SNR']:.2f}")

if __name__ == "__main__":
    main()

import json
import torch
import argparse
import csv
import numpy as np
from time import time, strftime
from pytorch_dedispersion.file_handler import FileHandler
from pytorch_dedispersion.data_processor import DataProcessor
from pytorch_dedispersion.dedispersion import Dedispersion
from pytorch_dedispersion.boxcar_filter import BoxcarFilter
from pytorch_dedispersion.candidate_finder import CandidateFinder

def load_config(config_file):
    """Load configuration from a JSON file."""
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def generate_dm_range(dm_ranges):
    """Generate a tensor of DM values based on specified ranges and steps."""
    dm_values = []
    for dm_range in dm_ranges:
        start = dm_range["start"]
        stop = dm_range["stop"]
        step = dm_range["step"]
        dm_values.extend(torch.arange(start, stop, step).tolist())
    return torch.tensor(dm_values)

def save_candidates_to_csv(candidates, filename):
    """Save candidate information to a CSV file."""
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SNR", "Sample Number", "Time (sec)", "Boxcar Width", "DM Value"])
        for candidate in candidates:
            writer.writerow([
                candidate['SNR'],
                candidate['Time Sample'],
                candidate['Time (sec)'],
                candidate['Boxcar Width'],
                candidate['DM Value']
            ])

def print_gpu_memory_usage(device, label=""):
    """Print the current GPU memory usage."""
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    summary = torch.cuda.memory_summary(device=device)
    print(f"{label}\nMemory Allocated: {allocated / (1024 ** 2):.2f} MB")
    print(f"Memory Reserved: {reserved / (1024 ** 2):.2f} MB")
    print(summary)


def get_total_gpu_memory():
    """Get the total GPU memory available."""
    total_memory = torch.cuda.get_device_properties(0).total_memory
    return total_memory

def load_bad_channels(file_path):
    """Load bad channel indices from a file."""
    with open(file_path, 'r') as file:
        content = file.read().strip()
        bad_channels = list(map(int, content.split()))
    return bad_channels

def dedisperse_and_find_candidates(config, verbose=False, remove_trend=False, window_size=None, gpu_index=0):
    """Perform dedispersion and find candidates."""
    if verbose:
        start_time = time()

    # Load file
    handler = FileHandler(config["SOURCE"])
    file_path = handler.load_file()
    if verbose:
        print(f"File loaded from: {file_path}")

    # Process data
    processor = DataProcessor(file_path)
    processor.load_data()
    data = processor.data
    frequencies = processor.get_frequencies()
    time_resolution = processor.header.tsamp
    if verbose:
        data_prep_time = time() - start_time
        print(f"Data prepared in {data_prep_time:.2f} seconds")
        start_time = time()

    # Load bad channels if specified
    bad_channels = []
    if "BAD_CHANNEL_FILE" in config:
        bad_channels = load_bad_channels(config["BAD_CHANNEL_FILE"])
        if verbose:
            print(f"Loaded bad channels: {bad_channels}")

    # Mask bad channels
    if bad_channels:
        data = np.delete(data, bad_channels, axis=0)
        frequencies = np.delete(frequencies, bad_channels)


    # Dedispersion parameters
    dm_range = generate_dm_range(config["DM_RANGES"])

    # Convert data to tensor
    device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    frequencies_tensor = torch.tensor(frequencies, dtype=torch.float32).to(device)
    freq_start = frequencies_tensor[0]
    dm_range = dm_range.to(device)

    # Print initial GPU memory usage
    if verbose and device.type == 'cuda':
        print_gpu_memory_usage(device, "Initial GPU memory usage:")

    # Calculate required memory for dedispersion
    initial_memory = torch.cuda.memory_allocated(device)
    num_dms = len(dm_range)
    total_gpu_memory = torch.cuda.get_device_properties(device).total_memory
    available_memory = total_gpu_memory * 0.90

    required_memory_per_dm = initial_memory * 4
    batch_fraction = available_memory / (required_memory_per_dm * num_dms)
    batch_size = int(batch_fraction * len(frequencies))
    if batch_size < 1:
        raise MemoryError("Batch size is less than 1 frequency channel. Reduce the number of DM trials to fit within available GPU memory.")

    summed_data = torch.zeros((len(dm_range), data.shape[1]), device=device)

    if batch_size < len(frequencies):
        # Perform dedispersion in batches
        print("Performing dedispersion in batches due to memory constraints...")
        for start_idx in range(0, len(frequencies), batch_size):
            end_idx = min(start_idx + batch_size, len(frequencies))
            freq_batch = frequencies_tensor[start_idx:end_idx]
            data_batch = data_tensor[start_idx:end_idx, :]

            dedisperse = Dedispersion(data_batch, freq_batch, dm_range, freq_start, time_resolution)
            dedispersed_data = dedisperse.perform_dedispersion()
            batch_summed_data = dedispersed_data.sum(dim=1)

            summed_data += batch_summed_data
            
            # Free up memory
            del freq_batch, data_batch, dedispersed_data, batch_summed_data
            torch.cuda.empty_cache()
    else:
        # Perform dedispersion without batching
        dedisperse = Dedispersion(data_tensor, frequencies_tensor, dm_range, freq_start, time_resolution)
        dedispersed_data = dedisperse.perform_dedispersion()
        summed_data = dedispersed_data.sum(dim=1)
    
    if verbose:
        dedispersion_time = time() - start_time
        print(f"Dedispersion and summing completed in {dedispersion_time:.2f} seconds")
        start_time = time()

    # Print GPU memory usage after dedispersion
    if verbose and device.type == 'cuda':
        print_gpu_memory_usage(device, "GPU memory usage after dedispersion:")

    # Boxcar filtering
    boxcar = BoxcarFilter(summed_data)
    boxcar_data = boxcar.apply_boxcar(config["BOXCAR_WIDTHS"])
    if verbose:
        boxcar_time = time() - start_time
        print(f"Boxcar filtering completed in {boxcar_time:.2f} seconds")
        start_time = time()

    # Find candidates
    finder = CandidateFinder(boxcar_data, window_size)
    candidates = finder.find_candidates(config["SNR_THRESHOLD"], config["BOXCAR_WIDTHS"], remove_trend)
    if verbose:
        candidate_finding_time = time() - start_time
        print(f"Candidate finding completed in {candidate_finding_time:.2f} seconds")

    # Enhance candidate data with actual DM values and time in seconds
    enhanced_candidates = []
    for candidate in candidates:
        dm_index = candidate['DM Index']
        dm_value = dm_range[dm_index].item()
        sample_number = candidate['Time Sample']
        time_in_sec = sample_number * time_resolution
        enhanced_candidates.append({
            'SNR': candidate['SNR'],
            'Time Sample': sample_number,
            'Time (sec)': time_in_sec,
            'Boxcar Width': candidate['Boxcar Width'],
            'DM Value': dm_value
        })

    # Generate CSV file with candidate information
    timestamp = strftime("%Y%m%d-%H%M%S")
    filename = f"candidates_{timestamp}.csv"
    save_candidates_to_csv(enhanced_candidates, filename)
    if verbose:
        print(f"Candidates saved to {filename}")


def main():
    """Dedisperse data and find candidates."""
    parser = argparse.ArgumentParser(description="Dedisperse data and find candidates.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--remove-trend", action="store_true", help="Remove trend from the data.")
    parser.add_argument("--window-size", type=int, help="Window size for trend removal (required if --remove-trend is specified).")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use (default: 0).")

    args = parser.parse_args()

    if args.remove_trend and not args.window_size:
        parser.error("--window-size is required when --remove-trend is specified.")

    config = load_config(args.config)

    dedisperse_and_find_candidates(
        config=config,
        verbose=args.verbose,
        remove_trend=args.remove_trend,
        window_size=args.window_size,
        gpu_index=args.gpu
    )


if __name__ == "__main__":
    main()

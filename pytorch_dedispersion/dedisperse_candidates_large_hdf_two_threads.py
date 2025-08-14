import json
import torch
import argparse
import csv
import numpy as np
from time import time, strftime
from tqdm import tqdm
from pytorch_dedispersion.file_handler import FileHandler
from pytorch_dedispersion.data_processor import DataProcessor
from pytorch_dedispersion.dedispersion import Dedispersion
from pytorch_dedispersion.boxcar_filter import BoxcarFilter
from pytorch_dedispersion.candidate_finder import CandidateFinder
import h5py
import os
from threading import Thread
from queue import Queue


def cpu_loader(file_path, total_freq, slice_size, bad_channels,
               out_q, pbar_scan, offset=0, step=1):
    """
    Worker thread: reads slices [offset::step].
    """
    for s in range(offset * slice_size,
                   total_freq,
                   step * slice_size):
        e = min(s + slice_size, total_freq)
        proc = DataProcessor(file_path,
                             freq_slice=(s, e),
                             bad_channels=bad_channels)
        proc.load_data()
        pbar_scan.update(1)

        if proc.data.shape[0] == 0:
            continue

        tensor_data = torch.as_tensor(proc.data,
                                      dtype=torch.float32).pin_memory()
        tensor_freq = torch.as_tensor(proc.get_frequencies(),
                                      dtype=torch.float32).pin_memory()
        out_q.put((tensor_data, tensor_freq))

    
    if offset == 0:
        out_q.put(None)

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def generate_dm_range(dm_ranges):
    dm_values = []
    for dm_range in dm_ranges:
        start = dm_range["start"]
        stop = dm_range["stop"]
        step = dm_range["step"]
        dm_values.extend(torch.arange(start, stop, step).tolist())
    return torch.tensor(dm_values)

def save_candidates_to_csv(candidates, filename):
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
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    summary = torch.cuda.memory_summary(device=device)
    print(f"{label}\nMemory Allocated: {allocated / (1024 ** 2):.2f} MB")
    print(f"Memory Reserved: {reserved / (1024 ** 2):.2f} MB")
    print(summary)

def save_dedispersed_data(original_file_path, summed_data, dm_range, tsamp, verbose=True):
    """
    Save the dedispersed and summed data to an HDF5
    Args:
        original_file_path (str): Path to the original input HDF5 file
        summed_data (torch.Tensor): The dedispersed data summed over frequency
        dm_range (torch.Tensor): Dispersion measure values corresponding to the data
        tsamp (float): Time sampling resolution in seconds
        verbose (bool): Whether to print status messages
    """
    # Extract base name from original file
    base_name = os.path.splitext(os.path.basename(original_file_path))[0]
    output_filename = f"dedispersed_{base_name}.h5"

    # Ensure no existing file conflicts
    if os.path.exists(output_filename):
        os.remove(output_filename)

    # Convert to CPU before saving
    summed_data_cpu = summed_data.cpu().numpy()
    dm_range_cpu = dm_range.cpu().numpy()

    with h5py.File(output_filename, 'w') as hf:
        hf.create_dataset(
            "summed_data", 
            data=summed_data_cpu, 
            compression="gzip", 
            compression_opts=9
        )
        hf.create_dataset(
            "dm_values", 
            data=dm_range_cpu, 
            compression="gzip", 
            compression_opts=9
        )
        # Store metadata
        hf.attrs["tsamp"] = tsamp
        hf.attrs["nDM"]   = summed_data.shape[0]
        hf.attrs["nTime"] = summed_data.shape[1]

    if verbose:
        print(f"Dedispersed data saved to {output_filename}")

def load_bad_channels(file_path):
    with open(file_path, 'r') as file:
        content = file.read().strip()
        bad_channels = list(map(int, content.split()))
    return bad_channels

def dedisperse_and_find_candidates(config, verbose=False, remove_trend=False, window_size=None, gpu_index=0):
    if verbose:
        overall_start = time()

    # load bad channels from config    
    bad_channels = []
    if "BAD_CHANNEL_FILE" in config:
        bad_channels = load_bad_channels(config["BAD_CHANNEL_FILE"])
        if verbose:
            print(f"Loaded bad channels from {config['BAD_CHANNEL_FILE']}")
            #print(f"Bad channels: {bad_channels}")

    handler = FileHandler(config["SOURCE"])
    file_path = handler.load_file()
    if verbose:
        print(f"File loaded from: {file_path}")

    # Check if we want to load the HDF5 data in frequency batches.
    if config.get("LOAD_FREQ_BATCH_SIZE", 0) > 0 and file_path.endswith('.hdf5'):
        load_freq_batch_size = config["LOAD_FREQ_BATCH_SIZE"]
        # Open file to get header info without loading full data.
        with h5py.File(file_path, 'r') as hdf:
            obs = hdf['Observation1']
            tuning = obs['Tuning1']
            total_freq = tuning['I'].shape[1]  # Original shape: (nsamples, nchans)
            time_resolution = obs.attrs.get('tInt', 0.001337)
            # Fetch the full frequency array and convert to MHz.
            full_frequencies = tuning['freq'][:] / 1e6  
            if full_frequencies[0] < full_frequencies[-1]:  # Ensure descending order
                full_frequencies = full_frequencies[::-1]
        freq_start = full_frequencies[0]
        dm_range = generate_dm_range(config["DM_RANGES"])
        device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
        dm_range = dm_range.to(device)
        global_summed_data = None
        nsamples = None

        # Loop over frequency slices using tqdm.
        total_slices = (total_freq + load_freq_batch_size - 1) // load_freq_batch_size
        # Progress bars: one for the disk scan, one for GPU work
        pbar_scan = tqdm(total=total_slices, desc="Scanning HDF5", position=0)
        pbar_gpu  = tqdm(desc="Dedispersing", position=1)

        #pbar = tqdm(total=total_slices, desc="Dedispersing slices")
        # Allocate a small queue (max 2 items) and start the loader thread
        q = Queue(maxsize=16)
        NUM_WORKERS = 2          # try 2 first; 3–4 if disk is NVMe

        for w in range(NUM_WORKERS):
            Thread(target=cpu_loader,
                args=(file_path, total_freq, load_freq_batch_size,
                     bad_channels, q, pbar_scan, w, NUM_WORKERS),daemon=True).start()

        copy_stream    = torch.cuda.Stream(device)
        compute_stream = torch.cuda.current_stream(device)

        # We don’t know nsamples until we’ve seen the first slice:
        first_item = q.get()           # blocks until loader has something
        if first_item is None:
            raise RuntimeError("All slices were empty after bad-channel masking.")

        data_np, freqs_np = first_item
        nsamples   = data_np.shape[1]
        max_chans  = load_freq_batch_size               # same as before

        # One reusable GPU buffer (size = one slice)
        gpu_buf = torch.empty((max_chans, nsamples),
                              dtype=torch.float32, device=device)
        freq_buf = torch.empty((max_chans,),
                               dtype=torch.float32, device=device)

        next_item = first_item          # we already have slice 0
        while next_item is not None:
            # ── kick off async H→D copy of slice n in copy_stream ──────────────
            data_np, freqs_np = next_item
            nchan_slice = data_np.shape[0]
            with torch.cuda.stream(copy_stream):
                #gpu_buf[:nchan_slice].copy_(torch.from_numpy(data_np),
                #                            non_blocking=True)
                #freq_buf[:nchan_slice].copy_(torch.from_numpy(freqs_np),
                #                             non_blocking=True)
                gpu_buf[:nchan_slice].copy_(data_np,  non_blocking=True)
                freq_buf[:nchan_slice].copy_(freqs_np, non_blocking=True)
           
           # ── meanwhile get slice n+1 ready on the CPU ───────────────────────
            next_item = q.get()   # may block, but overlaps with GPU copy

            # ── wait for copy to finish, then run kernels in compute_stream ────
            compute_stream.wait_stream(copy_stream)
            data_tensor = gpu_buf[:nchan_slice]
            freq_tensor = freq_buf[:nchan_slice]

            dedisperser = Dedispersion(data_tensor, freq_tensor,
                                       dm_range, freq_start, time_resolution)
            with torch.cuda.stream(compute_stream):
                dedispersed = dedisperser.perform_dedispersion()
                batch_sum   = dedispersed.sum(dim=1)
                if global_summed_data is None:
                    global_summed_data = batch_sum
                else:
                    global_summed_data += batch_sum
            pbar_gpu.update(1)

        pbar_gpu.close()
        pbar_scan.close()

        summed_data = global_summed_data
        tsamp = time_resolution  # Use the time resolution from the file.
    else:
        # Normal processing: load the full dataset.
        processor = DataProcessor(file_path, bad_channels=bad_channels)
        processor.load_data()
        data = processor.data
        frequencies = processor.get_frequencies()
        time_resolution = processor.header.tsamp
        tsamp = time_resolution
        dm_range = generate_dm_range(config["DM_RANGES"])
        device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
        frequencies_tensor = torch.tensor(frequencies, dtype=torch.float32).to(device)
        freq_start = frequencies_tensor[0]
        dm_range = dm_range.to(device)
        if device.type == "cuda":
            total_gpu_memory = torch.cuda.get_device_properties(device).total_memory
            available_memory = total_gpu_memory * 0.90
        else:
            available_memory = None
        required_memory_per_frequency = data.shape[1] * 4 * len(dm_range) * 4
        if available_memory is not None:
            batch_size = int(available_memory / required_memory_per_frequency)
        else:
            batch_size = data.shape[0]
        summed_data = torch.zeros((len(dm_range), data.shape[1]), device=device)
        if batch_size < len(frequencies):
            print("Performing dedispersion in batches due to memory constraints...")
            total_batches = len(range(0, len(frequencies), batch_size))
            for start_idx in tqdm(range(0, len(frequencies), batch_size), total=total_batches, desc="Processing batches"):
                end_idx = min(start_idx + batch_size, len(frequencies))
                freq_batch = frequencies_tensor[start_idx:end_idx]
                data_batch = torch.tensor(data[start_idx:end_idx, :], dtype=torch.float32).to(device)
                dedisperse_obj = Dedispersion(data_batch, freq_batch, dm_range, freq_start, time_resolution)
                dedispersed_data = dedisperse_obj.perform_dedispersion()
                batch_summed_data = dedispersed_data.sum(dim=1)
                summed_data += batch_summed_data
                del freq_batch, data_batch, dedisperse_obj, dedispersed_data, batch_summed_data
                torch.cuda.empty_cache()
        else:
            data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
            dedisperse_obj = Dedispersion(data_tensor, frequencies_tensor, dm_range, freq_start, time_resolution)
            dedispersed_data = dedisperse_obj.perform_dedispersion()
            summed_data = dedispersed_data.sum(dim=1)
        tsamp = time_resolution

    # Save the dedispersed and summed data before proceeding
    save_dedispersed_data(file_path, summed_data, dm_range, tsamp, verbose=verbose)

    if verbose:
        print("Dedispersion and summing completed.")

    # Boxcar filtering and candidate detection.
    if "BOXCAR_BATCH_SIZE" in config and config["BOXCAR_BATCH_SIZE"] > 0 and config["BOXCAR_BATCH_SIZE"] < len(config["BOXCAR_WIDTHS"]):
        candidates = []
        batch_size = config["BOXCAR_BATCH_SIZE"]
        for i in tqdm(range(0, len(config["BOXCAR_WIDTHS"]), batch_size), 
                  total=(len(config["BOXCAR_WIDTHS"]) + batch_size - 1) // batch_size, 
                  desc="Applying Boxcar Filtering"):
            widths_batch = config["BOXCAR_WIDTHS"][i:i+batch_size]
            boxcar = BoxcarFilter(summed_data)
            boxcar_data = boxcar.apply_boxcar(widths_batch)
            finder = CandidateFinder(boxcar_data, window_size)
            batch_candidates = finder.find_candidates(config["SNR_THRESHOLD"], widths_batch, remove_trend)
            candidates.extend(batch_candidates)
    else:
        boxcar = BoxcarFilter(summed_data)
        boxcar_data = boxcar.apply_boxcar(config["BOXCAR_WIDTHS"])
        finder = CandidateFinder(boxcar_data, window_size)
        candidates = finder.find_candidates(config["SNR_THRESHOLD"], config["BOXCAR_WIDTHS"], remove_trend)

    if verbose:
        print("Boxcar filtering and candidate finding completed.")

    enhanced_candidates = []
    for candidate in candidates:
        dm_index = candidate['DM Index']
        dm_value = dm_range[dm_index].item()
        sample_number = candidate['Time Sample']
        time_in_sec = sample_number * tsamp
        enhanced_candidates.append({
            'SNR': candidate['SNR'],
            'Time Sample': sample_number,
            'Time (sec)': time_in_sec,
            'Boxcar Width': candidate['Boxcar Width'],
            'DM Value': dm_value
        })

    timestamp = strftime("%Y%m%d-%H%M%S")
    filename = f"candidates_{timestamp}.csv"
    save_candidates_to_csv(enhanced_candidates, filename)
    if verbose:
        total_time = time() - overall_start
        print(f"Candidates saved to {filename}")
        print(f"Total processing time: {total_time:.2f} seconds")

def main():
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

import numpy as np
import json
from typing import List, Tuple

def compute_octave_bins(num_bins: int, freq_min: float = 20, freq_max: float = 20000) -> List[Tuple[float, float]]:
    """
    Computes the frequency ranges for octave bins.
    """
    bins = []
    log_min = np.log10(freq_min)
    log_max = np.log10(freq_max)
    for i in range(num_bins):
        start_freq = np.power(10, log_min + i * (log_max - log_min) / num_bins)
        end_freq = np.power(10, log_min + (i + 1) * (log_max - log_min) / num_bins)
        bins.append((start_freq, end_freq))
    return bins

def print_octave_bins(num_bins_list, freq_min=86, freq_max=20000):
    for num_bins in num_bins_list:
        print(f"\nNumber of Bins: {num_bins}")
        bins = compute_octave_bins(num_bins, freq_min, freq_max)
        for i, (start, end) in enumerate(bins):
            print(f"Bin {i + 1}: {start:.2f} Hz - {end:.2f} Hz")

def create_json(num_bins_list, freq_min=86, freq_max=20000):
    bins_dict = {}
    for num_bins in num_bins_list:
        bins = compute_octave_bins(num_bins, freq_min, freq_max)
        bins_dict[num_bins] = [(int(start), int(end)) for start, end in bins]
    return bins_dict

def write_json(bins_dict, filename="octave_bins.json"):
    with open(filename, "w") as f:
        json.dump(bins_dict, f, indent=4)

if __name__ == "__main__":
    num_bins_list = [8, 16, 32, 64]
    print_octave_bins(num_bins_list)
    bins_dict = create_json(num_bins_list)
    write_json(bins_dict)


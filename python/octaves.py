import numpy as np

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

def print_octave_bins(num_bins_list, freq_min=20, freq_max=20000):
    for num_bins in num_bins_list:
        print(f"\nNumber of Bins: {num_bins}")
        bins = compute_octave_bins(num_bins, freq_min, freq_max)
        for i, (start, end) in enumerate(bins):
            print(f"Bin {i + 1}: {start:.2f} Hz - {end:.2f} Hz")

if __name__ == "__main__":
    num_bins_list = [8, 16, 32, 64]
    print_octave_bins(num_bins_list)

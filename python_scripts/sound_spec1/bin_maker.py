import numpy as np
from typing import List, Tuple

SAMPLE_RATE = 44100
NYQUIST = 44100 / 2
FFT_SIZE = 512
MIN_FREQ = SAMPLE_RATE / FFT_SIZE # 86.13
MAX_FREQ = SAMPLE_RATE / 2 # 22050





def create_equal_bins(num_bins: int, min_freq: float = MIN_FREQ, max_freq: float = MAX_FREQ) -> List[Tuple[float, float]]:
    
    if num_bins not in [8, 16, 32, 64]:
        raise ValueError("Number of bins must be 8, 16, 32, or 64")
    
    bins = []

    start_freq = 0
    first_bin_max = max_freq / num_bins

    for _ in range(num_bins):
        end_freq = start_freq + first_bin_max
        print(f"Start: {start_freq} End: {end_freq}")
        # bins.append((start_freq, end_freq))
        start_freq = end_freq
     
create_equal_bins(8) # 8 bins

def create_frequency_bins(num_bins: int, min_freq: float = MIN_FREQ, max_freq: float = MAX_FREQ) -> List[Tuple[float, float]]:
    if num_bins not in [8, 16, 32, 64]:
        raise ValueError("Number of bins must be 8, 16, 32, or 64")
    
    # Generate the FFT frequency bins
    freqs = np.linspace(0, MAX_FREQ, FFT_SIZE // 2 + 1)
    # Filter out frequencies below min_freq
    freqs = freqs[freqs >= min_freq]
    
    # Calculate bin edges
    bin_edges = np.linspace(freqs[0], freqs[-1], num_bins + 1)
    bins = []

    for i in range(num_bins):
        start_freq = bin_edges[i]
        end_freq = bin_edges[i + 1]
        bins.append((start_freq, end_freq))
        print(f"Bin {i + 1}: Start: {start_freq:.2f} Hz End: {end_freq:.2f} Hz")
    
    return bins

create_frequency_bins(8) # 8 bins
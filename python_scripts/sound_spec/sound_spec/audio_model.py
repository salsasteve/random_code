import numpy as np

# import soundfile as sf
import logging
from scipy.fft import fft
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

### Model ###

SAMPLE_RATE = 44100
NYQUIST_FREQUENCY = SAMPLE_RATE / 2
FFT_SIZE = 512
POSITIVE_FREQS = FFT_SIZE // 2
FREQUENCY_RESOLUTION = SAMPLE_RATE / FFT_SIZE  # 86.13
MAX_FREQ = SAMPLE_RATE / 2  # 22050
MAX_HUMAN_HEARING_FREQ = 20000
DEFAULT_NUM_BINS = 8


class AudioModel:
    def __init__(
        self,
        signal: np.ndarray,
        fft_size: int = FFT_SIZE,
        num_bins: int = DEFAULT_NUM_BINS,
        sample_rate: int = SAMPLE_RATE,
        alpha: float = 0.3,
        bin_split: str = "octave",
    ):
        if num_bins not in [8, 16, 32, 64]:
            raise ValueError("Number of bins must be 8, 16, 32, or 64")

        if alpha <= 0 or alpha > 1:
            raise ValueError("Smoothing factor must be in the range (0, 1]")

        self.signal = signal
        self.samplerate = sample_rate
        self.fft_size = fft_size
        self.position = 0
        self.window = np.hamming(fft_size)
        self.prev_bins = None  # For smoothing
        self.alpha = alpha  # Smoothing factor
        if bin_split == "linear":
            self.bin_config = self.create_linear_bins_config(num_bins=num_bins)
        else:
            self.bin_config = self.create_octave_bins_config(num_bins=num_bins)
        self.bin_indexes = self.get_bin_indexes()
        self.num_bins = num_bins

    # def read_wav(self, filename):
    #     signal, samplerate = sf.read(filename, dtype="int16")
    #     # Normalize if necessary
    #     if signal.dtype != np.float32:
    #         signal = signal / np.max(np.abs(signal))
    #     return signal, samplerate

    def get_next_chunk(self) -> np.ndarray:
        """
        Returns the next chunk of audio data from the signal.
        """
        start = self.position
        end = start + self.fft_size

        if start >= len(self.signal):
            return None  # End of signal
        elif end > len(self.signal):
            # Pad the chunk if it's shorter than fft_size
            chunk = self.signal[start:]
            chunk = np.pad(chunk, (0, self.fft_size - len(chunk)), "constant")
            self.position = len(self.signal)
        else:
            chunk = self.signal[start:end]
            self.position = end

        # Apply window function
        chunk = chunk * self.window
        return chunk

    def compute_fft(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the FFT of a given audio chunk.
        """
        yf = np.abs(fft(frame))[:POSITIVE_FREQS]
        # yf are the magnitudes of the FFT
        xf = np.fft.fftfreq(FFT_SIZE, 1 / self.samplerate)[:POSITIVE_FREQS]
        # xf is the frequency axis
        return xf, yf

    def smooth_bins(self, bins: np.ndarray) -> np.ndarray:
        if self.prev_bins is None:
            self.prev_bins = bins
            return bins
        else:
            smoothed_bins = self.alpha * bins + (1 - self.alpha) * self.prev_bins
            self.prev_bins = smoothed_bins
            return smoothed_bins

    def scale_bins(
        self, bins: np.ndarray, scale_min: int = 0, scale_max: int = 65
    ) -> np.ndarray:
        """On the display we have 64 pixels, so we scale the bins to fit in this range."""
        max_bin = np.max(bins)

        # Avoid division by zero by adding a small epsilon value
        epsilon = 1e-6
        scaled_bins = scale_max * bins / (max_bin + epsilon)

        scaled_bins = np.clip(scaled_bins, scale_min, scale_max)

        return scaled_bins.astype(int)
    
    def create_linear_bins_config(
        self,
        num_bins: int = DEFAULT_NUM_BINS,
        min_freq: float = FREQUENCY_RESOLUTION,
        max_freq: float = MAX_HUMAN_HEARING_FREQ,
    ) -> List[Tuple[float, float]]:
        """
        Computes frequency ranges for bins with linear spacing.
        """
        bins = []
        bin_width = (max_freq - min_freq) / num_bins
        for i in range(num_bins):
            start_freq = min_freq + i * bin_width
            end_freq = start_freq + bin_width
            bins.append((start_freq, end_freq))
        return bins

    def create_octave_bins_config(
        self,
        num_bins: int = DEFAULT_NUM_BINS,
        frequency_resolution: float = FREQUENCY_RESOLUTION,
        max_freq: float = MAX_HUMAN_HEARING_FREQ,
        bin_size_multiplier: float = 3.0,
    ) -> List[Tuple[float, float]]:
        """
        Computes the frequency ranges for octave bins with adjustable bin sizes.
        """
        bins = []
        log_min = np.log10(frequency_resolution)
        log_max = np.log10(max_freq)
        total_log_range = log_max - log_min

        bin_range = total_log_range / num_bins
        for i in range(num_bins):
            start_freq = 10 ** (log_min + i * bin_range)
            end_freq = 10 ** (log_min + (i + 1) * bin_range)
            bins.append((start_freq, end_freq))
            
        return bins

    def get_bin_indexes(self) -> List[Tuple[int, int]]:
        """
        Returns the indexes of the bins in the FFT.
        """
        bin_indexes = []
        for start, end in self.bin_config:
            start_idx = int(start / FREQUENCY_RESOLUTION)
            end_idx = int(end / FREQUENCY_RESOLUTION)
            bin_indexes.append((start_idx, end_idx))
        return bin_indexes

    def sort_and_average_bins(self, yf: np.ndarray) -> np.ndarray:
        """
        Sorts the FFT bins into octave bins and averages the values.
        """
        bins = np.zeros(self.num_bins)

        for i, (start, end) in enumerate(self.bin_indexes):
            if start == end:
                bins[i] = yf[start]
            else:
                bins[i] = np.mean(yf[start:end])
        return bins

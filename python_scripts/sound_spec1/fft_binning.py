
import numpy as np
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import logging
import threading
from scipy.fft import fft
from matplotlib.animation import FuncAnimation
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

### Model ###

SAMPLE_RATE = 44100
NYQUIST = SAMPLE_RATE / 2
FFT_SIZE = 512
MIN_FREQ = SAMPLE_RATE / FFT_SIZE # 86.13
MAX_FREQ = SAMPLE_RATE / 2 # 22050
MAX_HUMAN_HEARING_FREQ = 20000
DEFAULT_NUM_BINS = 8

class AudioModel:
    def __init__(self, filename: str, fft_size: int = FFT_SIZE, num_bins: int = DEFAULT_NUM_BINS):

        if num_bins not in [8, 16, 32, 64]:
            raise ValueError("Number of bins must be 8, 16, 32, or 64")
        
        self.signal, self.samplerate = self.read_wav(filename)
        self.chunk_size = fft_size
        self.position = 0
        self.window = np.hamming(fft_size)
        self.prev_bins = None  # For smoothing
        self.alpha = 0.3       # Smoothing factor (0 < alpha <= 1)
        self.bins = self.create_octave_bins()

    def read_wav(self, filename):
        signal, samplerate = sf.read(filename, dtype='int16')
        # Normalize if necessary
        if signal.dtype != np.float32:
            signal = signal / np.max(np.abs(signal))
        return signal, samplerate

    def get_next_chunk(self):
        start = self.position
        end = start + self.chunk_size

        if start >= len(self.signal):
            return None  # End of signal
        elif end > len(self.signal):
            # Pad the chunk if it's shorter than chunk_size
            chunk = self.signal[start:]
            chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), 'constant')
            self.position = len(self.signal)
        else:
            chunk = self.signal[start:end]
            self.position = end

        # Apply window function
        chunk = chunk * self.window
        return chunk

    def compute_fft(self, chunk):
        """
        Computes the FFT of a given audio chunk.
        """
        # yf is the magnitude of the FFT
        yf = np.abs(fft(chunk))[:len(chunk) // 2]
        # xf is the frequency axis
        xf = np.fft.fftfreq(len(chunk), 1 / self.samplerate)[:len(chunk) // 2]
        return xf, yf

    def smooth_bins(self, bins):
        if self.prev_bins is None:
            self.prev_bins = bins
            return bins
        else:
            smoothed_bins = self.alpha * bins + (1 - self.alpha) * self.prev_bins
            self.prev_bins = smoothed_bins
            return smoothed_bins

    def scale_bins(self, bins, scale_min=0, scale_max=63): 
        """On the display we have 64 pixels, so we scale the bins to fit in this range."""
        min_bin = np.min(bins)
        max_bin = np.max(bins)
        scaled_bins = scale_max * (bins - min_bin) / (max_bin - min_bin + 1e-6)
        return np.clip(scaled_bins, scale_min, scale_max)
    
    def create_octave_bins(self, num_bins: int = DEFAULT_NUM_BINS, min_freq: float = MIN_FREQ, max_freq: float = MAX_HUMAN_HEARING_FREQ) -> List[Tuple[float, float]]:
        """
        Computes the frequency ranges for octave bins.
        """
        bins = []
        log_min = np.log10(min_freq)
        log_max = np.log10(max_freq)
        total_freq_range = log_max - log_min
        bin_range = total_freq_range / num_bins
        for i in range(num_bins):
            start_freq = np.power(10, log_min + i * bin_range)  
            end_freq = np.power(10, log_min + (i + 1) * bin_range)
            bins.append((start_freq, end_freq))
        return bins

### View ###

class VisualizerView:
    def __init__(self, model, visualization_mode='bar', num_bins=8, freq_min=86,freq_max=20000):
        self.model = model
        self.visualization_mode = visualization_mode
        self.bin_ranges = self.model.create_octave_bins(num_bins, freq_min, freq_max)
        self.num_bins = num_bins
        self.fig = None
        self.ax = None
        self.ani = None

    def init_bar_plot(self, freq_ranges):
        num_bins = len(freq_ranges)
        self.fig, self.ax = plt.subplots()
        self.bars = self.ax.bar(range(num_bins), np.zeros(num_bins), color=plt.cm.rainbow(np.linspace(0, 1, num_bins)))
        self.ax.set_ylim(0, 63)

        # Set x-axis ticks and labels
        x_positions = np.arange(num_bins)
        freq_labels = [f"{int(start_freq)}-{int(end_freq)}" for start_freq, end_freq in freq_ranges]
        self.ax.set_xticks(x_positions)
        self.ax.set_xticklabels(freq_labels, rotation=45, ha='right')
        self.ax.set_xlabel('Frequency Range (Hz)')
        self.ax.set_ylabel('Amplitude')

    def init_line_plot(self):
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'b-')
        self.ax.set_xlim(0, self.model.samplerate / 2)
        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.set_ylabel('Amplitude')

    def update_bar_plot(self, frame):
        chunk = self.model.get_next_chunk()
        if chunk is None:
            self.ani.event_source.stop()
            return self.bars,

        xf, yf = self.model.compute_fft(chunk)
        freq_ranges = self.model.get_frequency_ranges(num_bins=self.num_bins)
        bins = self.model.create_logarithmic_bins(xf, yf, freq_ranges)
        bins = self.model.smooth_bins(bins)
        scaled_bins = self.model.scale_bins(bins)

        for bar, height in zip(self.bars, scaled_bins):
            bar.set_height(height)
        return self.bars,

    def update_line_plot(self, frame):
        chunk = self.model.get_next_chunk()
        if chunk is None:
            self.ani.event_source.stop()
            return self.line,

        xf, yf = self.model.compute_fft(chunk)
        yf = self.model.smooth_bins(yf)
        self.line.set_data(xf, yf)
        self.ax.relim()
        self.ax.autoscale_view()
        return self.line,

    def animate(self):
        interval = (self.model.chunk_size / self.model.samplerate) * 1000  # in milliseconds

        if self.visualization_mode == 'bar':
            freq_ranges = self.model.get_frequency_ranges(num_bins=self.num_bins)
            self.init_bar_plot(freq_ranges)
            self.ani = FuncAnimation(self.fig, self.update_bar_plot, init_func=lambda: (self.bars,), interval=interval, blit=False)
        elif self.visualization_mode == 'line':
            self.init_line_plot()
            self.ani = FuncAnimation(self.fig, self.update_line_plot, init_func=lambda: (self.line,), interval=interval, blit=False)
        else:
            raise ValueError(f"Invalid visualization_mode: {self.visualization_mode}")

        plt.show()

### Controller ###

def play_audio(signal, samplerate):
    """
    Plays the audio signal.
    """
    sd.play(signal, samplerate)
    sd.wait()  # Wait until playback is finished

def main():
    filename = "PinkPanther60.wav"
    visualization_mode = 'bar'

    model = AudioModel(filename=filename)

    # test model

    
    # view = VisualizerView(model, visualization_mode=visualization_mode, num_bins=num_bins)

    # # Start audio playback in a separate thread
    # play_thread = threading.Thread(target=play_audio, args=(model.signal, model.samplerate))
    # play_thread.start()

    # # Start the visualization
    # view.animate()

    # # Wait for the audio playback to finish
    # play_thread.join()

if __name__ == "__main__":
    main()

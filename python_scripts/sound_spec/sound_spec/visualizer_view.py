import numpy as np
import matplotlib.pyplot as plt
import logging
from matplotlib.animation import FuncAnimation
from sound_spec.audio_model import AudioModel, NYQUIST_FREQUENCY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualizerView:
    def __init__(self, model: AudioModel):
        self.model = model
        self.num_bins = model.num_bins
        self.bin_config = model.bin_config
        self.fig = None
        self.ax = None
        self.ani = None

    def init_bar_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))

        self.bars = self.ax.bar(
            range(self.num_bins),
            np.zeros(self.num_bins),
            color=plt.cm.rainbow(np.linspace(0, 1, self.num_bins)),
        )
        self.ax.set_ylim(0, 64)

        # Set x-axis ticks and labels
        x_positions = np.arange(self.num_bins)
        freq_labels = [
            f"{int(start_freq)}-{int(end_freq)}"
            for start_freq, end_freq in self.bin_config
        ]
        self.ax.set_xticks(x_positions)
        self.ax.set_xticklabels(freq_labels, rotation=45, ha="right")
        self.ax.set_xlabel("Frequency Range (Hz)")
        self.ax.set_ylabel("Amplitude")
        plt.tight_layout()

    def init_line_plot(self):
        self.fig, self.ax = plt.subplots()
        (self.line,) = self.ax.plot([], [], "b-")
        self.ax.set_xlim(0, NYQUIST_FREQUENCY)
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Amplitude")

    def update_bar_plot(self, frame):
        chunk = self.model.get_next_chunk()
        if chunk is None:
            self.ani.event_source.stop()
            return (self.bars,)

        _, yf = self.model.compute_fft(chunk)

        bins = self.model.sort_and_average_bins(yf)
        bins = self.model.smooth_bins(bins)
        scaled_bins = self.model.scale_bins(bins)

        for bar, height in zip(self.bars, scaled_bins):
            bar.set_height(height)
        return (self.bars,)

    def animate(self) -> FuncAnimation:
        interval = (
            self.model.fft_size / self.model.samplerate
        ) * 1000  # in milliseconds

        self.init_bar_plot()
        self.ani = FuncAnimation(
            self.fig,
            self.update_bar_plot,
            init_func=lambda: (self.bars,),
            interval=interval,
            blit=False,
        )
        plt.show()
        return self.ani

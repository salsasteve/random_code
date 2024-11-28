import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from sound_spec.audio_model import AudioModel
import threading
import queue
import time

# Parameters
SAMPLE_RATE = 44100
FFT_SIZE = 1024
NUM_BINS = 8

# PyAudio Parameters
CHUNK = FFT_SIZE  # Number of frames per buffer
FORMAT = pyaudio.paInt16
CHANNELS = 2  # Stereo recording
DEVICE_INDEX = 0  # Set this to the correct device index

# Initialize AudioModel
audio_model = AudioModel(
    signal=None,  # We will feed real-time data
    fft_size=FFT_SIZE,
    num_bins=NUM_BINS,
    sample_rate=SAMPLE_RATE,
    alpha=0.3,
    bin_split="linear"
)

# Create a queue to share data between threads
data_queue = queue.Queue()

def convert_audio_data(in_data):
    """Convert byte data to numpy array and normalize."""
    audio_data = np.frombuffer(in_data, dtype=np.int16)
    normalized_data = audio_data.astype(np.float32) / 32768.0
    print(f"Converted audio data: {normalized_data[:10]}")  # Print first 10 samples for debugging
    return normalized_data

def process_audio_chunk(audio_model, audio_data):
    """Process audio chunk and return scaled bins."""
    audio_model.signal = audio_data
    audio_model.position = 0  # Reset position

    chunk = audio_model.get_next_chunk()
    if chunk is not None:
        xf, yf = audio_model.compute_fft(chunk)
        print(f"FFT result (yf): {yf[:10]}")  # Print first 10 FFT results for debugging
        bins = audio_model.sort_and_average_bins(yf)
        print(f"Sorted and averaged bins: {bins}")  # Print bins for debugging
        smoothed_bins = audio_model.smooth_bins(bins)
        print(f"Smoothed bins: {smoothed_bins}")  # Print smoothed bins for debugging
        scaled_bins = audio_model.scale_bins(smoothed_bins)
        print(f"Scaled bins: {scaled_bins}")  # Print scaled bins for debugging
        return scaled_bins
    return None

def audio_callback(in_data, frame_count, time_info, status):
    """This callback is called for each audio block."""
    if status:
        print(f"Status: {status}")

    audio_data = convert_audio_data(in_data)
    scaled_bins = process_audio_chunk(audio_model, audio_data)

    if scaled_bins is not None:
        data_queue.put(scaled_bins)

    return (in_data, pyaudio.paContinue)

def audio_stream():
    """Initialize and start the audio stream."""
    pa = pyaudio.PyAudio()

    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=DEVICE_INDEX,
        frames_per_buffer=CHUNK,
        stream_callback=audio_callback
    )

    stream.start_stream()

    while stream.is_active():
        time.sleep(0.1)

    stream.stop_stream()
    stream.close()
    pa.terminate()

def main():
    """Main function to start audio processing and plotting."""
    audio_thread = threading.Thread(target=audio_stream)
    audio_thread.start()

    print("Listening... Press Ctrl+C to stop.")

    plt.figure()
    bar_container = plt.bar(range(NUM_BINS), [0]*NUM_BINS)
    plt.ylim(0, 65)
    plt.xlabel('Frequency Bin')
    plt.ylabel('Amplitude')
    plt.title('Real-Time Audio Spectrum')
    plt.pause(0.01)

    current_bins = np.zeros(NUM_BINS)

    try:
        while True:
            if not data_queue.empty():
                scaled_bins = data_queue.get()
                print(f"Scaled bins from queue: {scaled_bins}")  # Print scaled bins from queue for debugging
                current_bins = scaled_bins
            else:
                # Decrement the bin values
                current_bins = np.maximum(current_bins - 1, 0)

            for rect, h in zip(bar_container.patches, current_bins):
                rect.set_height(h)
            plt.draw()
            plt.pause(0.01)
    except KeyboardInterrupt:
        print("Stopping...")

if __name__ == "__main__":
    main()
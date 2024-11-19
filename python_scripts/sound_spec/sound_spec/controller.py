import numpy as np
import soundfile as sf
import sounddevice as sd
import logging
import threading
from sound_spec.audio_model import AudioModel
from sound_spec.visualizer_view import VisualizerView

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def play_audio(signal, samplerate):
    """
    Plays the audio signal.
    """
    sd.play(signal, samplerate)
    sd.wait()  # Wait until playback is finished

def read_wav(filename):
    signal, samplerate = sf.read(filename, dtype='int16')
    # Normalize if necessary
    if signal.dtype != np.float32:
        signal = signal / np.max(np.abs(signal))
    return signal, samplerate


def main():
    filename = "python_scripts\sound_spec\sounds\chirp.wav"

    signal, samplerate = read_wav(filename)

    model = AudioModel(signal, sample_rate=samplerate)

    view = VisualizerView(model)

    # Start audio playback in a separate thread
    play_thread = threading.Thread(target=play_audio, args=(model.signal, model.samplerate))
    play_thread.start()

    # Start the visualization
    view.animate()

    # Wait for the audio playback to finish
    play_thread.join()

if __name__ == "__main__":
    main()
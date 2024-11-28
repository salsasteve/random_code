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
    signal, samplerate = sf.read(filename, dtype="int16")
    # Normalize if necessary
    if signal.dtype != np.float32:
        signal = signal / np.max(np.abs(signal))
    return signal, samplerate

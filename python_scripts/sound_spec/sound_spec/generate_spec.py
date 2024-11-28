import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf


def generate_chirp(
    bits: int = 16,
    duration: int = 5,
    chirplength: int = 3,
    samplerate: int = 44100,
    start_freq: int = 20,
    end_freq: int = 20000,
):
    # Time arrays for chirp and constant frequency
    t_chirp = np.linspace(0, chirplength, int(chirplength * samplerate), endpoint=False)
    t_const = np.linspace(
        chirplength,
        duration,
        int((duration - chirplength) * samplerate),
        endpoint=False,
    )

    # Generate chirp frequencies from f0 to f1 over chirplength
    freqs_chirp = np.linspace(start_freq, end_freq, len(t_chirp))
    # Constant frequency f1 for the remaining duration
    freqs_const = np.full(len(t_const), end_freq)

    # Concatenate frequencies and time arrays
    freqs = np.concatenate((freqs_chirp, freqs_const))

    # Compute the phase by integrating the frequency
    phase = np.cumsum(freqs) * (2 * np.pi / samplerate)
    signal = np.sin(phase)
    if bits == 16:
        signal = np.int16(signal * 2**15)
    elif bits == 32:
        signal = np.int32(signal * 2**31)

    return signal


def display_signal(signal):
    print(signal.shape)
    plt.plot(signal[0 : int(signal.shape[0] * 0.04)])
    plt.show()


def save_signal(signal, filename, samplerate: int = 44100):
    sf.write(filename, signal, samplerate)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

# Sample rate and duration
fs = 1000  # Sample rate in Hz
duration = 4  # Total duration in seconds

# Time vector
t = np.linspace(0, duration, fs * duration, endpoint=False)

# Create non-stationary signal: 50 Hz for 1 sec, 100 Hz for 1 sec, 250 Hz for 2 sec
signal = np.piecewise(
    t,
    [t < 1, (t >= 1) & (t < 2), t >= 2],
    [lambda t: np.sin(2 * np.pi * 50 * t),
     lambda t: np.sin(2 * np.pi * 100 * t),
     lambda t: np.sin(2 * np.pi * 250 * t)]
)

# STFT parameters
window = 'hann'  # Window function
nperseg = 256    # Window length (in samples)
noverlap = nperseg // 2  # 50% overlap

# Compute STFT
f, t_stft, Zxx = stft(signal, fs, window=window, nperseg=nperseg, noverlap=noverlap)

# Plot STFT magnitude
plt.figure(figsize=(10, 6))
plt.pcolormesh(t_stft, f, np.abs(Zxx), shading='gouraud')
plt.title("STFT Magnitude")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [sec]")
plt.colorbar(label="Magnitude")
plt.show()

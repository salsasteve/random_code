import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

# Sample rate and duration
fs = 8000  # Sample rate in Hz
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

# STFT parameters for ~15.62 Hz frequency resolution and 0.016 s temporal resolution
window = 'hann'     # Window function
nperseg =  512      # Window length for ~15.62 Hz frequency resolution
noverlap = 384       # 48-sample overlap for ~0.016 s temporal resolution

# Compute STFT
f, t_stft, Zxx = stft(signal, fs, window=window, nperseg=nperseg, noverlap=noverlap)

# Verifying frequency resolution
frequency_resolution = fs / nperseg
print(f"Frequency Resolution: {frequency_resolution} Hz")

# Verifying temporal resolution
temporal_resolution = (nperseg - noverlap) / fs
print(f"Temporal Resolution: {temporal_resolution} seconds")

# Plot STFT magnitude
plt.figure(figsize=(10, 6))
plt.pcolormesh(t_stft, f, np.abs(Zxx), shading='gouraud')
plt.title("STFT Magnitude with Desired Frequency and Temporal Resolution")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [sec]")
plt.colorbar(label="Magnitude")
plt.ylim(0, 300)  # Limiting frequency axis for better visualization
plt.show()

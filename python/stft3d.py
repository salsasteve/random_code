import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from mpl_toolkits.mplot3d import Axes3D

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
window = 'hann'    # Window function
nperseg = 64       # Window length for ~15.62 Hz frequency resolution
noverlap = 48      # 48-sample overlap for ~0.016 s temporal resolution

# Compute STFT
f, t_stft, Zxx = stft(signal, fs, window=window, nperseg=nperseg, noverlap=noverlap)

# Prepare data for 3D plot
T, F = np.meshgrid(t_stft, f)
Z = np.abs(Zxx)  # Magnitude of STFT

# Plot in 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(T, F, Z, cmap='viridis')

# Labeling
ax.set_title("3D STFT Magnitude")
ax.set_xlabel("Time [sec]")
ax.set_ylabel("Frequency [Hz]")
ax.set_zlabel("Magnitude")
fig.colorbar(surf, ax=ax, label="Magnitude")
plt.show()

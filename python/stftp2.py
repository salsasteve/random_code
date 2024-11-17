import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
from scipy.signal import stft

fs = 10000

duration = 1

t = np.linspace(0, duration, int(fs * duration) + 1)

n1 = np.arange(150, 2001) - 1
n2 = np.arange(3500, 6001) - 1
n3 = np.arange(7000, 9001) - 1

signal = np.zeros_like(t)

signal[n1] = tukey(len(n1), alpha=0.5) * np.cos(2 * np.pi * 25 * t[n1])
signal[n2] = tukey(len(n2), alpha=0.25) * np.cos(2 * np.pi * 125 * t[n2])
signal[n3] = tukey(len(n3), alpha=0.75) * np.cos(2 * np.pi * 64 * t[n3])

# plt.figure(figsize=(10, 4))
# plt.plot(t, signal)
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Non-Stationary Signal')
# plt.grid(True)
# plt.show()


window = 'hamming'    
nperseg = 512    
noverlap = 384     

f, t_stft, Zxx = stft(signal, fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=512)

freq_limit = 200
freq_indices = np.where(f <= freq_limit)[0]
f = f[freq_indices]
Zxx = Zxx[freq_indices, :]
T, F = np.meshgrid(t_stft, f)
Z = np.abs(Zxx)


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(T, F, Z, cmap='viridis')

ax.set_title("3D STFT Magnitude Window size 512 and 384 overlap and 512 FFT")
ax.set_xlabel("Time [sec]")
ax.set_ylabel("Frequency [Hz]")
ax.set_zlabel("Magnitude")
fig.colorbar(surf, ax=ax, label="Magnitude")
plt.show()



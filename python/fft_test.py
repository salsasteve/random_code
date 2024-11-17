import numpy as np
import time

# Example signals
x = np.random.random(65536)
h = np.random.random(65536)

# Measure time for direct convolution
start_time = time.time()
y_direct = np.convolve(x, h, mode='full')
direct_conv_time = time.time() - start_time

# Measure time for FFT-based method
start_time = time.time()
X = np.fft.fft(x, len(x) + len(h) - 1)
H = np.fft.fft(h, len(x) + len(h) - 1)
Y = X * H
y_fft = np.fft.ifft(Y).real
fft_conv_time = time.time() - start_time

# Verify results
if np.allclose(y_direct, y_fft):
    print("The results are the same!")
else:
    print("The results differ.")

print(f"Direct convolution time: {direct_conv_time:.6f} seconds")
print(f"FFT-based convolution time: {fft_conv_time:.6f} seconds")
print(f"Difference: {np.max(np.abs(y_direct - y_fft))}")


import matplotlib.pyplot as plt
import numpy as np
import json
from math import log10

# Set frequency range
min_freq = 20        # Minimum frequency (20 Hz)
max_freq = 20000     # Maximum frequency (20,000 Hz)

# Parameters to adjust
num_bins = 8         # Number of bins (adjust as needed)
exponent = 1      # Adjust to control bin distribution (only for logarithmic scale)
scale_type = 'linear'   # Choose 'log' for logarithmic scale or 'linear' for linear scale

# Generate bin edges
bin_edges = []

if scale_type == 'log':
    
    for i in range(num_bins + 1):
        freq = 10 ** (log10(min_freq) + i * (log10(max_freq) - log10(min_freq)) / num_bins)
        
        bin_edges.append(freq)
elif scale_type == 'linear':    
    bin_edges = np.linspace(min_freq, max_freq, num_bins + 1)
else:
    raise ValueError("Invalid scale_type. Choose 'log' or 'linear'.")

bins = []
for i in range(num_bins):
    start_freq = bin_edges[i]
    end_freq = bin_edges[i + 1]
    bins.append({'start_freq': start_freq, 'end_freq': end_freq})
    print(f"Bin {i + 1}: {start_freq:.2f} Hz to {end_freq:.2f} Hz")

# Prepare data for plotting
x = np.linspace(min_freq, max_freq, 1000)
y = np.ones_like(x)

# Plot the bins
plt.figure(figsize=(12, 6))
plt.plot(x, y, label='Frequency Range')
for edge in bin_edges:
    plt.axvline(x=edge, color='r', linestyle='--')

plt.xscale(scale_type)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title(f'Frequency Bins ({scale_type.capitalize()} Scale)')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.show()

# Display bin ranges
# bins = []
# for i in range(num_bins):
#     start_freq = bin_edges[i]
#     end_freq = bin_edges[i + 1]
#     bins.append({'start_freq': start_freq, 'end_freq': end_freq})
#     print(f"Bin {i + 1}: {start_freq:.2f} Hz to {end_freq:.2f} Hz")

# Allow manual adjustment
print("\nAdjust the 'num_bins', 'exponent', and 'scale_type' variables in the code to change the bin distribution.")
approve = input("Do you approve these bin ranges? (y/n): ")

if approve.lower() == 'y':
    # Output bins as JSON
    with open('bins2.json', 'w') as json_file:
        json.dump(bins, json_file, indent=4)
    print("Bin ranges saved to 'bins.json'.")
else:
    print("Bin ranges not saved. Please adjust the parameters and run again.")
# Sound Spec

**Creator:** Elman Steve Laguna

## Introduction

Sound Spec is a Music visualizer that can take in wav and realtime audio as input.

## Project Breakdown

Theres a couple of parts to this project:
- AudioModel: 
    - Takes in input for the type of data that will be processed
        - signal: 16 bit signed integer array
    - Initializes the audio model with the following parameters:
        - SAMPLE_RATE: The sample rate of the audio(44100)
        - FFT_SIZE: The size of the fft(512)
        - POSITIVE_FREQS: FFT_SIZE // 2 (half of the fft size)
        - FREQUENCY_RESOLUTION: SAMPLE_RATE / FFT_SIZE (86.13)
        - MAX_FREQ: SAMPLE_RATE / 2 (22050)
        - MAX_HUMAN_HEARING_FREQ: 20000
        - DEFAULT_NUM_BINS: 8
    - Creates bins at initialization
    - Creates chunks of audio data during runtime
    - Has functions to smooth, scale, and sort the bins
- Visualizer:
    - Takes in the audio model and plots the audio data in real time
    - Uses matplotlib to plot the data
    - Uses the audio model to get the next chunk and compute the fft
    - Uses the audio model to smooth, scale, and sort the bins

- Chirp Generator:
    - Generates a chirp signal
    - Takes in input for creating multiple chirps representing different bins
    

- Controllers:
    - File Based Controller:
        - Initializes the audio model
        - Initializes the visualizer
        - Reads the audio file
        - Starts the audio stream
        - Starts the visualizer
        - Starts the audio stream and visualizer in separate threads
    - RealtimeController(IN PROGRESS):
        - Initializes the audio model
        - Initializes the visualizer
        - Starts the audio stream
        - Starts the visualizer
        - Starts the audio stream and visualizer in separate threads


### 1. Project Setup

- **Environment Setup:**
  - Describe the tools and libraries used (e.g., Python, NumPy, SciPy, Matplotlib, PyAudio).
  - Explain how to set up the development environment.

- **Directory Structure:**
  - Provide an overview of the project's directory structure.
  - Example:
    ```
    sound_spec/
    ├── audio_model.py
    ├── examples/
    │   └── realtime.py
    ├── tests/
    │   └── test_audio_model.py
    ├── .venv/
    └── README.md
    ```

### 2. Audio Model

- **Initialization:**
  - Explain the `AudioModel` class and its initialization parameters.
  - Example:
    ```python
    audio_model = AudioModel(
        signal=None,
        fft_size=1024,
        num_bins=8,
        sample_rate=44100,
        alpha=0.3,
        bin_split="linear"
    )
    ```

- **Methods:**
  - Describe the key methods in the `AudioModel` class:
    - `get_next_chunk()`
    - `compute_fft()`
    - `smooth_bins()`
    - `scale_bins()`
    - `create_linear_bins_config()`
    - `create_octave_bins_config()`
    - `get_bin_indexes()`
    - `sort_and_average_bins()`

### 3. Real-Time Audio Processing

- **Audio Callback:**
  - Explain the `audio_callback` function and its role in processing audio data in real-time.

- **Audio Stream:**
  - Describe the `audio_stream` function and how it initializes and starts the audio stream.

- **Main Function:**
  - Provide an overview of the `main` function, which starts the audio processing and plotting.

### 4. Debugging and Testing

- **Debugging:**
  - Describe the debugging process and the use of print statements to log intermediate results.

- **Unit Testing:**
  - Explain how to create unit tests for the `AudioModel` class and its methods.

### 5. Visualization

- **Plotting:**
  - Describe how Matplotlib is used to visualize the real-time audio spectrum.
  - Example:
    ```python
    plt.figure()
    bar_container = plt.bar(range(NUM_BINS), [0]*NUM_BINS)
    plt.ylim(0, 65)
    plt.xlabel('Frequency Bin')
    plt.ylabel('Amplitude')
    plt.title('Real-Time Audio Spectrum')
    plt.pause(0.01)
    ```

### 6. Conclusion

- Summarize the key points of the project.
- Discuss potential future improvements and features.

## References

- List any references or resources used in the creation of the Sound Spec library.
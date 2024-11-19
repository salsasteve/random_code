import pytest
import numpy as np
from sound_spec.audio_model import (
    NYQUIST_FREQUENCY,
    AudioModel,
    SAMPLE_RATE,
    FFT_SIZE,
    DEFAULT_NUM_BINS,
    MAX_FREQ,
    FREQUENCY_RESOLUTION,
)
from unittest.mock import patch, MagicMock


@pytest.fixture(scope="module")
def sample_signal():
    # Create a synthetic audio signal (e.g., a sine wave at 440 Hz)
    duration = 1  # Duration in seconds
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    frequency = 440  # Frequency in Hz
    test_signal = np.sin(2 * np.pi * frequency * t)
    # test signal has 44100 samples
    return test_signal


@pytest.fixture
def audio_model(sample_signal):
    # Initialize the AudioModel with the synthetic signal
    model = AudioModel(
        signal=sample_signal,
        fft_size=FFT_SIZE,
        num_bins=DEFAULT_NUM_BINS,
        sample_rate=SAMPLE_RATE,
    )
    return model

def test_audio_model_bad_num_bins(sample_signal):
    with pytest.raises(ValueError):
        AudioModel(
            signal=sample_signal,
            fft_size=FFT_SIZE,
            num_bins=10,
            sample_rate=SAMPLE_RATE,
        )

def test_audio_model_bad_alpha(sample_signal):
    with pytest.raises(ValueError):
        AudioModel(
            signal=sample_signal,
            fft_size=FFT_SIZE,
            num_bins=DEFAULT_NUM_BINS,
            sample_rate=SAMPLE_RATE,
            alpha=1.5,
        )



def test_get_next_chunk(audio_model):
    chunk = audio_model.get_next_chunk()
    assert chunk is not None
    assert len(chunk) == FFT_SIZE


def test_get_next_chunk_end_of_signal(audio_model):
    # Set the position to the end of the signal
    audio_model.position = len(audio_model.signal)
    chunk = audio_model.get_next_chunk()
    assert chunk is None


def test_get_next_chunk_padding(audio_model):
    # Set the position to the end of the signal
    audio_model.position = len(audio_model.signal) - 10
    chunk = audio_model.get_next_chunk()
    assert len(chunk) == FFT_SIZE


def test_compute_fft(audio_model):
    chunk = audio_model.get_next_chunk()
    xf, yf = audio_model.compute_fft(chunk)
    assert len(xf) == FFT_SIZE // 2
    assert len(yf) == FFT_SIZE // 2


def test_compute_fft_xf_output(audio_model):
    chunk = audio_model.get_next_chunk()
    xf, _ = audio_model.compute_fft(chunk)
    expected_last_freq = NYQUIST_FREQUENCY - FREQUENCY_RESOLUTION
    # 22050 = 44100 / 2 (NYQUIST_FREQUENCY)
    # 86.13 = 44100 / 512 (FREQUENCY_RESOLUTION)
    # 21963.86 = 22050 - 86.13 (expected_last_freq)
    assert xf[0] == 0
    assert pytest.approx(xf[-1], abs=1e-6) == expected_last_freq


def test_compute_fft_yf_output(audio_model):
    chunk = audio_model.get_next_chunk()
    _, yf = audio_model.compute_fft(chunk)
    # A 440 Hz sine wave should produce a peak at 440 Hz in the FFT
    # This peak is located at index 440 / 86.13 = 5.1
    # We will check the values at indices 5 and 6

    # Calculate the expected index for 440 Hz
    expected_index = int(440 / (SAMPLE_RATE / FFT_SIZE))

    peak_value = yf[expected_index]
    assert (
        peak_value > 0.5
    ), f"Expected peak at index {expected_index} to be greater than 0.5, but got {peak_value}"

    # Check that the values around the peak are lower
    for i in range(1, 4):
        assert (
            yf[expected_index - i] < peak_value
        ), f"Expected value at index {expected_index - i} to be less than peak value"
        assert (
            yf[expected_index + i] < peak_value
        ), f"Expected value at index {expected_index + i} to be less than peak value"


def test_smooth_bins_first_call(audio_model):
    bins = np.array([1.0, 2.0, 3.0])
    smoothed_bins = audio_model.smooth_bins(bins)
    assert np.array_equal(smoothed_bins, bins)


def test_smooth_bins_second_call(audio_model):
    bins = np.array([1.0, 2.0, 3.0])
    smoothed_bins_first = audio_model.smooth_bins(bins)
    smoothed_bins_second = audio_model.smooth_bins(smoothed_bins_first)
    # Calculate the expected smoothed values
    alpha = audio_model.alpha
    expected_smoothed_bins = alpha * smoothed_bins_first + (1 - alpha) * bins
    assert np.allclose(
        smoothed_bins_second, expected_smoothed_bins
    ), f"Expected {expected_smoothed_bins}, but got {smoothed_bins_second}"


def test_scale_bins(audio_model):
    bins = np.array([0.0, 0.5, 1.0])
    scaled_bins = audio_model.scale_bins(bins)
    assert scaled_bins[0] == 0
    assert scaled_bins[1] == 32
    assert scaled_bins[2] == 64

def test_create_octave_bins(audio_model):
    bins = audio_model.create_octave_bins()
    assert len(bins) == DEFAULT_NUM_BINS
    for start, end in bins:
        assert start < end


if __name__ == "__main__":
    pytest.main()

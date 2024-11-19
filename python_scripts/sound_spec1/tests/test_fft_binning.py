
import pytest
import numpy as np
from audio_model import AudioModel, SAMPLE_RATE, FFT_SIZE, DEFAULT_NUM_BINS

@pytest.fixture
def audio_model():
    # Use a small array of synthetic audio data for testing
    test_signal = np.sin(2 * np.pi * np.linspace(0, 1, SAMPLE_RATE))
    model = AudioModel(filename="test.wav")
    model.signal = test_signal
    model.samplerate = SAMPLE_RATE
    return model

def test_read_wav(audio_model):
    signal, samplerate = audio_model.read_wav("test.wav")
    assert len(signal) > 0
    assert samplerate == SAMPLE_RATE

# def test_get_next_chunk(audio_model):
#     chunk = audio_model.get_next_chunk()
#     assert chunk is not None
#     assert len(chunk) == FFT_SIZE

# def test_compute_fft(audio_model):
#     chunk = audio_model.get_next_chunk()
#     xf, yf = audio_model.compute_fft(chunk)
#     assert len(xf) > 0
#     assert len(yf) > 0

# def test_smooth_bins(audio_model):
#     bins = np.array([1.0, 2.0, 3.0])
#     smoothed_bins = audio_model.smooth_bins(bins)
#     assert np.array_equal(smoothed_bins, bins)

# def test_scale_bins(audio_model):
#     bins = np.array([0.0, 0.5, 1.0])
#     scaled_bins = audio_model.scale_bins(bins)
#     assert all(0 <= x <= 63 for x in scaled_bins)

# def test_create_octave_bins(audio_model):
#     bins = audio_model.create_octave_bins()
#     assert len(bins) == DEFAULT_NUM_BINS
#     for start, end in bins:
#         assert start < end

if __name__ == "__main__":
    pytest.main()

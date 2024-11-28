from sound_spec.audio_model import AudioModel
from sound_spec.visualizer_view import VisualizerView
from sound_spec.controller import play_audio, read_wav
import threading


def main():
    filename = r"C:\Users\Steve\Documents\code\random_code\python_scripts\sound_spec\sounds\8_15005.0_17502.5.wav"

    signal, samplerate = read_wav(filename)

    model = AudioModel(signal, sample_rate=samplerate, bin_split="linear")

    view = VisualizerView(model)

    # Start audio playback in a separate thread
    play_thread = threading.Thread(
        target=play_audio, args=(model.signal, model.samplerate)
    )
    play_thread.start()

    # Start the visualization
    view.animate()

    # Wait for the audio playback to finish
    play_thread.join()


if __name__ == "__main__":
    main()

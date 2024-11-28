from sound_spec.audio_model import AudioModel
from sound_spec.controller import read_wav



def main():
    filename = r"C:\Users\Steve\Documents\code\random_code\python_scripts\sound_spec\sounds\8_85_169.wav"

    signal, samplerate = read_wav(filename)

    model = AudioModel(signal, sample_rate=samplerate)

    print(model.bin_config)


if __name__ == "__main__":
    main()
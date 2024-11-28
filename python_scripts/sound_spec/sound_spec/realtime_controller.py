import sys
import pyaudio

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1 if sys.platform == 'darwin' else 2
RATE = 44100
RECORD_SECONDS = 5


def list_devices(p): 
    print("Available audio devices:") 
    for i in range(p.get_device_count()): 
        info = p.get_device_info_by_index(i) 
        print(f"{i}: {info['name']}") 
        p = pyaudio.PyAudio() 


def main():
    pass

if __name__ == "__main__":
    main()
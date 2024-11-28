import argparse
import json
from sound_spec.generate_spec import generate_chirp, save_signal

def main():
    parser = argparse.ArgumentParser(description='Generate chirp signals.')
    parser.add_argument('--start_freq', type=float, help='Starting frequency of the chirp.')
    parser.add_argument('--end_freq', type=float, help='Ending frequency of the chirp.')
    parser.add_argument('--duration', type=float, default=3.0, help='Total duration of the signal in seconds.')
    parser.add_argument('--chirp_length', type=float, default=3.0, help='Duration of the chirp within the signal.')
    parser.add_argument('--output', type=str, default='chirp.wav', help='Output filename for the generated chirp.')
    parser.add_argument('--bins_file', type=str, help='Path to JSON file containing frequency bins.')

    args = parser.parse_args()

    if args.start_freq and args.end_freq:
        # Generate a single chirp with provided frequencies
        start = args.start_freq
        end = args.end_freq
        print(f"Generating chirp from {start} Hz to {end} Hz")
        signal = generate_chirp(
            duration=args.duration,
            chirplength=args.chirp_length,
            start_freq=start,
            end_freq=end
        )
        save_signal(
            signal,
            args.output
        )
        print(f"Chirp saved to {args.output}")
    elif args.bins_file:
        # Generate chirps from a bins file
        with open(args.bins_file, 'r') as myfile:
            bins = json.load(myfile)

        for bin_item in bins:
            start = float(bin_item['start_freq'])
            end = float(bin_item['end_freq'])
            print(f"Generating chirp from {start} Hz to {end} Hz")
            signal = generate_chirp(
                duration=args.duration,
                chirplength=args.chirp_length,
                start_freq=start,
                end_freq=end
            )
            output_filename = f"chirp_{int(start)}_{int(end)}.wav"
            save_signal(
                signal,
                output_filename
            )
            print(f"Chirp saved to {output_filename}")
    else:
        print("Please provide either start and end frequencies or a bins file.")
        parser.print_help()

if __name__ == "__main__":
    main()

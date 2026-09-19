import os
import subprocess
import tempfile

import librosa
import numpy as np


INPUT_DIR = "dataset"
OUTPUT_DIR = "spectrograms"

SAMPLE_RATE = 22050
SEGMENT_DURATION = 5
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)


def convert_to_wav(input_file, output_file):
    command = [
        "ffmpeg",
        "-y",
        "-i", input_file,
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        output_file
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )


def process_song(input_file):
    song_name = os.path.splitext(
        os.path.basename(input_file)
    )[0]

    song_output_dir = os.path.join(
        OUTPUT_DIR,
        song_name
    )

    os.makedirs(song_output_dir, exist_ok=True)

    print("\nProcessing:", song_name)

    with tempfile.TemporaryDirectory() as temp_dir:

        wav_file = os.path.join(
            temp_dir,
            "audio.wav"
        )

        print("  Converting MP3 → WAV...")

        convert_to_wav(
            input_file,
            wav_file
        )

        print("  Loading audio...")

        audio, sr = librosa.load(
            wav_file,
            sr=SAMPLE_RATE,
            mono=True
        )

        total_duration = len(audio) / sr

        print(
            f"  Duration: {total_duration:.2f} seconds"
        )

        segment_samples = int(
            SEGMENT_DURATION * sr
        )

        segment_count = 0

        for start in range(
            0,
            len(audio) - segment_samples + 1,
            segment_samples
        ):

            end = start + segment_samples

            segment = audio[start:end]

            mel = librosa.feature.melspectrogram(
                y=segment,
                sr=sr,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS
            )

            mel_db = librosa.power_to_db(
                mel,
                ref=np.max
            )

            output_file = os.path.join(
                song_output_dir,
                f"segment_{segment_count:04d}.npy"
            )

            np.save(
                output_file,
                mel_db.astype(np.float32)
            )

            segment_count += 1

        print(
            f"  Created {segment_count} segments"
        )


def main():

    mp3_files = sorted(
        file for file in os.listdir(INPUT_DIR)
        if file.lower().endswith(".mp3")
        and ":Zone.Identifier" not in file
    )

    if not mp3_files:
        print("No MP3 files found.")
        return

    print("=" * 60)
    print("AudioVision Dataset Builder")
    print("=" * 60)

    print(f"Found {len(mp3_files)} MP3 files")

    for filename in mp3_files:

        input_file = os.path.join(
            INPUT_DIR,
            filename
        )

        try:
            process_song(input_file)

        except Exception as error:
            print(
                f"  ERROR processing {filename}: {error}"
            )

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

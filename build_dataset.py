import os
import subprocess
import numpy as np
import csv

REAL_DIR = "dataset/real"
FAKE_DIR = "dataset/fake"
OUTPUT = "dataset_features.csv"


def extract_features(audio):
    cmd = [
        "ffmpeg",
        "-i", audio,
        "-f", "f32le",
        "-ac", "1",
        "-ar", "16000",
        "-"
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    samples = np.frombuffer(result.stdout, dtype=np.float32)

    if len(samples) == 0:
        return None

    # Basic features
    duration = len(samples) / 16000

    rms = np.sqrt(np.mean(samples ** 2))

    peak = np.max(np.abs(samples))

    # Zero crossing rate
    zero_crossings = np.sum(
        np.abs(np.diff(np.sign(samples))) > 0
    )

    zcr = zero_crossings / len(samples)

    # Frequency spectrum
    spectrum = np.abs(np.fft.rfft(samples))

    frequencies = np.fft.rfftfreq(
        len(samples),
        1 / 16000
    )

    total_energy = np.sum(spectrum) + 1e-10

    spectral_centroid = (
        np.sum(frequencies * spectrum)
        / total_energy
    )

    # Dominant frequency
    dominant_frequency = frequencies[
        np.argmax(spectrum)
    ]

    return [
        duration,
        rms,
        peak,
        zcr,
        spectral_centroid,
        dominant_frequency
    ]


rows = []

print("=" * 50)
print("       🛡️ VOXGUARD AI")
print("     ML DATASET BUILDER")
print("=" * 50)


for label, folder in [(0, REAL_DIR), (1, FAKE_DIR)]:

    print(f"\n📁 Processing: {folder}")

    for filename in os.listdir(folder):

        path = os.path.join(folder, filename)

        features = extract_features(path)

        if features is not None:

            rows.append(
                [filename, label] + features
            )

            if label == 0:
                print("🟢 REAL:", filename)
            else:
                print("🔴 AI/Fake:", filename)


with open(OUTPUT, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "filename",
        "label",
        "duration",
        "rms",
        "peak",
        "zcr",
        "spectral_centroid",
        "dominant_frequency"
    ])

    writer.writerows(rows)


print("\n" + "=" * 50)
print("✅ DATASET CREATED")
print("=" * 50)

print("Total samples:", len(rows))
print("Output file:", OUTPUT)


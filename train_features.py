import os
import subprocess
import numpy as np
from scipy.fft import rfft

REAL_DIR = "dataset/real"
FAKE_DIR = "dataset/fake"


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

    rms = np.sqrt(np.mean(samples ** 2))
    peak = np.max(np.abs(samples))

    spectrum = np.abs(rfft(samples))
    dominant_frequency = (
        np.argmax(spectrum) * 16000 / len(samples)
    )

    duration = len(samples) / 16000

    return [
        duration,
        rms,
        peak,
        dominant_frequency
    ]


print("🛡️ VOXGUARD AI")
print("Feature Dataset Builder")
print("-" * 40)

for label, folder in [(0, REAL_DIR), (1, FAKE_DIR)]:

    print(f"\n📁 Processing: {folder}")

    for filename in os.listdir(folder):

        path = os.path.join(folder, filename)

        features = extract_features(path)

        if features is not None:
            print(f"✅ {filename}")
            print(f"   Features: {features}")
            print(f"   Label: {label}")

print("\n🎯 Feature extraction completed!")


import subprocess
import numpy as np
from scipy.fft import rfft

print("🛡️ VOXGUARD AI")
print("Audio Feature Extraction")
print("-" * 35)

audio = input("Enter audio file path: ").strip()

# Convert audio to raw mono PCM
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
    print("❌ Could not read audio.")
    exit()

# Basic audio features
duration = len(samples) / 16000
rms = np.sqrt(np.mean(samples ** 2))
peak = np.max(np.abs(samples))

# Frequency spectrum
spectrum = np.abs(rfft(samples))

dominant_frequency = np.argmax(spectrum) * 16000 / len(samples)

print("\n📊 AUDIO FEATURES")
print("-" * 35)
print(f"Duration: {duration:.2f} seconds")
print(f"Samples: {len(samples)}")
print(f"RMS Energy: {rms:.6f}")
print(f"Peak Amplitude: {peak:.6f}")
print(f"Dominant Frequency: {dominant_frequency:.2f} Hz")

print("\n✅ Feature extraction complete!")

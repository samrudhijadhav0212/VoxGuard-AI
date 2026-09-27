import numpy as np
import subprocess
import os

MODEL = "voxguard_model.npz"

def extract_features(audio):
    cmd = [
        "ffmpeg", "-i", audio,
        "-ac", "1",
        "-ar", "16000",
        "-f", "f32le",
        "-"
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    audio_data = np.frombuffer(result.stdout, dtype=np.float32)

    if len(audio_data) == 0:
        raise ValueError("Could not read audio")

    duration = len(audio_data) / 16000

    rms = np.sqrt(np.mean(audio_data ** 2))
    peak = np.max(np.abs(audio_data))

    zcr = np.mean(
        np.abs(np.diff(np.sign(audio_data)))
    ) / 2

    spectrum = np.abs(np.fft.rfft(audio_data))
    freqs = np.fft.rfftfreq(len(audio_data), 1 / 16000)

    spectral_centroid = (
        np.sum(freqs * spectrum) /
        (np.sum(spectrum) + 1e-8)
    )

    dominant_frequency = freqs[np.argmax(spectrum)]

    return np.array([
        duration,
        rms,
        peak,
        zcr,
        spectral_centroid,
        dominant_frequency
    ])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


print("\n🛡️ VoxGuard AI Voice Detection")
print("--------------------------------")

if not os.path.exists(MODEL):
    print("❌ Model file not found!")
    print("Make sure voxguard_model.npz is in ~/VoxGuard")
    exit()

audio = input("🎵 Enter audio file path: ").strip()

if not os.path.exists(audio):
    print("❌ Audio file not found!")
    exit()

data = np.load(MODEL)

weights = data["weights"]
bias = data["bias"]
mean = data["mean"]
std = data["std"]

features = extract_features(audio)

normalized = (features - mean) / (std + 1e-8)

score = np.dot(normalized, weights) + bias

ai_probability = sigmoid(score)

if ai_probability >= 0.75:
    prediction = "AI-GENERATED"
    risk = "HIGH"
elif ai_probability >= 0.50:
    prediction = "AI-GENERATED"
    risk = "MEDIUM"
else:
    prediction = "REAL / HUMAN"
    risk = "LOW"

confidence = max(ai_probability, 1 - ai_probability) * 100

print("\n========== VOXGUARD RESULT ==========")
print("Prediction :", prediction)
print(f"Confidence : {confidence:.2f}%")
print(f"AI Score   : {ai_probability * 100:.2f}%")
print("Risk Level :", risk)

print("\n📊 Audio Features")
print(f"Duration           : {features[0]:.2f} sec")
print(f"RMS                : {features[1]:.6f}")
print(f"Peak               : {features[2]:.6f}")
print(f"Zero Crossing Rate : {features[3]:.6f}")
print(f"Spectral Centroid  : {features[4]:.2f} Hz")
print(f"Dominant Frequency : {features[5]:.2f} Hz")

print("======================================")

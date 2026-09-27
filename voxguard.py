import subprocess
import os

print("=" * 40)
print("       🛡️ VOXGUARD AI")
print("   Detect. Verify. Protect.")
print("=" * 40)

audio = input("\nEnter audio file path: ").strip()

if not os.path.exists(audio):
    print("\n❌ Audio file not found.")
    exit()

print("\n🔍 Analyzing audio...")

# Get audio information
command = [
    "ffprobe",
    "-v", "error",
    "-show_entries",
    "format=duration,size",
    "-of", "default=noprint_wrappers=1",
    audio
]

result = subprocess.run(
    command,
    capture_output=True,
    text=True
)

print("\n📊 AUDIO INFORMATION")
print("-" * 30)
print(result.stdout)

# Create spectrogram
spectrum = os.path.expanduser(
    "~/storage/shared/Download/voxguard_spectrum.png"
)

print("\n🎵 Generating audio spectrum...")

cmd = [
    "ffmpeg",
    "-y",
    "-i", audio,
    "-lavfi",
    "showspectrumpic=s=800x400:legend=disabled",
    "-frames:v", "1",
    spectrum
]

subprocess.run(cmd)

print("\n✅ Spectrum generated!")
print("📁 Saved at:")
print(spectrum)

print("\n🛡️ VOXGUARD STATUS")
print("-" * 30)
print("Audio received successfully.")
print("Audio preprocessing: Complete")
print("Spectrum analysis: Complete")
print("AI Detection Model: Next Step")
print("Prediction: Pending ML Model")
print("Risk Score: Pending ML Model")

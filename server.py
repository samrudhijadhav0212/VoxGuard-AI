from http.server import HTTPServer, BaseHTTPRequestHandler
import subprocess
import numpy as np
import os
import re
import html

MODEL = "voxguard_model.npz"
PORT = 8000


def extract_features(path):
    cmd = [
        "ffmpeg", "-i", path,
        "-ac", "1", "-ar", "16000",
        "-f", "f32le", "-"
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    audio = np.frombuffer(result.stdout, dtype=np.float32)

    if len(audio) == 0:
        raise ValueError("Could not read audio")

    duration = len(audio) / 16000
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)

    spectrum = np.abs(np.fft.rfft(audio))
    freq = np.fft.rfftfreq(len(audio), 1 / 16000)

    centroid = float(
        np.sum(freq * spectrum) /
        (np.sum(spectrum) + 1e-8)
    )

    dominant = float(freq[np.argmax(spectrum)])

    return np.array([
        duration,
        rms,
        peak,
        zcr,
        centroid,
        dominant
    ])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def analyze(path):

    features = extract_features(path)

    model = np.load(MODEL)

    weights = model["weights"]
    bias = model["bias"]
    mean = model["mean"]
    std = model["std"]

    x = (features - mean) / (std + 1e-8)

    probability = float(
        sigmoid(np.dot(x, weights) + bias)
    )

    ai_score = probability * 100

    if probability >= 0.75:
        prediction = "AI-GENERATED"
        risk = "HIGH"
        explanation = "The voice shows patterns associated with the AI-generated samples used by the prototype."

    elif probability >= 0.50:
        prediction = "POTENTIALLY AI-GENERATED"
        risk = "MEDIUM"
        explanation = "Some analyzed patterns are associated with AI-generated samples. Further verification is recommended."

    else:
        prediction = "REAL / HUMAN"
        risk = "LOW"
        explanation = "The analyzed voice is closer to the human-voice patterns learned by the prototype."

    confidence = ai_score if probability >= 0.50 else 100 - ai_score

    return {
        "prediction": prediction,
        "confidence": confidence,
        "ai_score": ai_score,
        "risk": risk,
        "explanation": explanation,
        "duration": features[0],
        "rms": features[1],
        "peak": features[2],
        "zcr": features[3],
        "centroid": features[4],
        "dominant": features[5]
    }


def page(result=None, error=""):

    result_html = ""

    if result:

        risk = result["risk"]

        result_html = f"""
        <div class="result">

            <h2>VoxGuard Analysis</h2>

            <div class="prediction">
                {html.escape(result["prediction"])}
            </div>

            <div class="confidence">
                Confidence: {result["confidence"]:.2f}%
            </div>

            <div class="score-title">
                AI Score: {result["ai_score"]:.2f}%
            </div>

            <div class="bar">
                <div class="fill"
                     style="width:{result["ai_score"]:.2f}%">
                </div>
            </div>

            <div class="risk {risk.lower()}">
                Risk Level: {risk}
            </div>

            <div class="explanation">
                <h3>Why this result?</h3>
                <p>{html.escape(result["explanation"])}</p>
            </div>

            <div class="features">

                <div>
                    <span>Duration</span>
                    <b>{result["duration"]:.2f} sec</b>
                </div>

                <div>
                    <span>RMS</span>
                    <b>{result["rms"]:.4f}</b>
                </div>

                <div>
                    <span>Peak</span>
                    <b>{result["peak"]:.4f}</b>
                </div>

                <div>
                    <span>ZCR</span>
                    <b>{result["zcr"]:.4f}</b>
                </div>

                <div>
                    <span>Spectral Centroid</span>
                    <b>{result["centroid"]:.2f} Hz</b>
                </div>

                <div>
                    <span>Dominant Frequency</span>
                    <b>{result["dominant"]:.2f} Hz</b>
                </div>

            </div>

        </div>
        """

    if error:
        result_html = f"""
        <div class="error">
            {html.escape(error)}
        </div>
        """

    return f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>VoxGuard AI</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    padding: 30px 15px;
    font-family: Arial, sans-serif;
    color: white;
    background:
    radial-gradient(circle at top,
    #172554,
    #050816 70%);
}}

.container {{
    max-width: 800px;
    margin: auto;
}}

header {{
    text-align: center;
    margin-bottom: 30px;
}}

.logo {{
    font-size: 42px;
    font-weight: bold;
}}

.logo span {{
    color: #38bdf8;
}}

.tagline {{
    color: #7dd3fc;
    font-size: 19px;
    margin-top: 8px;
}}

.subtitle {{
    color: #94a3b8;
    margin-top: 10px;
}}

.card,
.result {{
    background: #111827;
    border: 1px solid #263653;
    border-radius: 22px;
    padding: 30px;
    margin-top: 20px;
    box-shadow: 0 15px 40px rgba(0,0,0,.3);
}}

.card {{
    text-align: center;
}}

.card h2 {{
    font-size: 27px;
}}

.upload {{
    border: 1px dashed #3b82f6;
    padding: 25px;
    border-radius: 15px;
    margin: 20px 0;
}}

input {{
    width: 100%;
    color: white;
}}

button {{
    padding: 14px 30px;
    border: none;
    border-radius: 10px;
    background: linear-gradient(90deg,#2563eb,#06b6d4);
    color: white;
    font-size: 17px;
    font-weight: bold;
}}

.prediction {{
    font-size: 32px;
    font-weight: bold;
    color: #38bdf8;
    text-align: center;
    margin: 20px 0;
}}

.confidence {{
    text-align: center;
    font-size: 18px;
}}

.score-title {{
    margin-top: 25px;
    margin-bottom: 8px;
}}

.bar {{
    height: 15px;
    background: #1e293b;
    border-radius: 20px;
    overflow: hidden;
}}

.fill {{
    height: 100%;
    background: linear-gradient(90deg,#2563eb,#22d3ee);
}}

.risk {{
    width: fit-content;
    margin: 20px auto;
    padding: 10px 25px;
    border-radius: 30px;
    font-weight: bold;
}}

.high {{
    background: #7f1d1d;
    color: #fecaca;
}}

.medium {{
    background: #78350f;
    color: #fde68a;
}}

.low {{
    background: #14532d;
    color: #bbf7d0;
}}

.explanation {{
    background: #0f172a;
    padding: 18px;
    border-radius: 14px;
    margin-top: 20px;
}}

.explanation h3 {{
    color: #38bdf8;
}}

.explanation p {{
    color: #cbd5e1;
    line-height: 1.6;
}}

.features {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 20px;
}}

.features div {{
    background: #0f172a;
    padding: 15px;
    border-radius: 12px;
}}

.features span {{
    display: block;
    color: #94a3b8;
    font-size: 13px;
    margin-bottom: 6px;
}}

.error {{
    background: #450a0a;
    color: #fecaca;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
}}

footer {{
    text-align: center;
    color: #64748b;
    margin-top: 30px;
    font-size: 13px;
}}

@media(max-width:600px) {{

    .features {{
        grid-template-columns: 1fr;
    }}

    .prediction {{
        font-size: 26px;
    }}

}}

</style>

</head>

<body>

<div class="container">

<header>

<div class="logo">
Vox<span>Guard</span> AI
</div>

<div class="tagline">
Detect · Verify · Protect
</div>

<div class="subtitle">
AI-Powered Voice Cloning Detection System
</div>

</header>

<div class="card">

<h2>🎙️ Analyze Voice</h2>

<p>
Upload an audio file to check whether it is
human or potentially AI-generated.
</p>

<form method="POST"
enctype="multipart/form-data">

<div class="upload">

<input type="file"
name="audio"
accept="audio/*"
required>

</div>

<button type="submit">
Analyze Voice
</button>

</form>

</div>

{result_html}

<footer>

VoxGuard AI · Deep Thinkers · SIH 2026

<br><br>

Prototype ML analysis — results depend on
audio quality and training data.

</footer>

</div>

</body>

</html>
"""


class VoxGuardHandler(BaseHTTPRequestHandler):

    def do_GET(self):

        data = page()

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "text/html; charset=utf-8"
        )

        self.end_headers()

        self.wfile.write(
            data.encode("utf-8")
        )


    def do_POST(self):

        try:

            content_type = self.headers.get(
                "Content-Type", ""
            )

            match = re.search(
                r"boundary=(.+)",
                content_type
            )

            if not match:
                raise ValueError(
                    "Upload boundary not found"
                )

            boundary = (
                "--" +
                match.group(1)
            ).encode()

            length = int(
                self.headers.get(
                    "Content-Length", 0
                )
            )

            body = self.rfile.read(length)

            parts = body.split(boundary)

            filename = None
            file_data = None

            for part in parts:

                if b'name="audio"' not in part:
                    continue

                header_end = part.find(
                    b"\r\n\r\n"
                )

                if header_end == -1:
                    continue

                headers = part[:header_end]

                data = part[
                    header_end + 4:
                ]

                found = re.search(
                    rb'filename="([^"]*)"',
                    headers
                )

                if found:

                    filename = found.group(1).decode(
                        "utf-8",
                        errors="ignore"
                    )

                    file_data = data

                    if file_data.endswith(
                        b"\r\n"
                    ):
                        file_data = file_data[:-2]

                    break

            if not filename or not file_data:
                raise ValueError(
                    "No audio file received"
                )

            filename = os.path.basename(filename)

            upload_path = "upload_" + filename

            with open(
                upload_path,
                "wb"
            ) as f:
                f.write(file_data)

            result = analyze(upload_path)

            try:
                os.remove(upload_path)
            except:
                pass

            data = page(result=result)

            self.send_response(200)

            self.send_header(
                "Content-Type",
                "text/html; charset=utf-8"
            )

            self.end_headers()

            self.wfile.write(
                data.encode("utf-8")
            )

        except Exception as e:

            data = page(error=str(e))

            self.send_response(500)

            self.send_header(
                "Content-Type",
                "text/html; charset=utf-8"
            )

            self.end_headers()

            self.wfile.write(
                data.encode("utf-8")
            )


print("===================================")
print("VOXGUARD AI SERVER")
print("===================================")
print("Open in browser:")
print("http://127.0.0.1:8000")
print("===================================")

server = HTTPServer(
    ("0.0.0.0", PORT),
    VoxGuardHandler
)

server.serve_forever()

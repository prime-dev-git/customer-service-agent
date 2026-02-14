import torch
import io
import numpy as np
from flask import Flask, request, send_file, jsonify
from kokoro import KPipeline
from pydub import AudioSegment
import soundfile as sf

app = Flask(__name__)

# 1. Initialize Kokoro (lang_code 'a' for American English)
# It will use GPU automatically if 'cuda' is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pipeline = KPipeline(lang_code='a', device=device)

@app.route('/v1/audio/speech', methods=['POST'])
def tts():
    try:
        data = request.json
        text = data.get("input", "")
        # Popular voices: af_heart, af_bella, am_adam, am_michael
        voice_name = data.get("voice", "af_heart") 

        if not text:
            return jsonify({"error": "No text"}), 400

        # 2. Generate Audio
        # Kokoro returns a generator of (graphemes, phonemes, audio_tensor)
        generator = pipeline(text, voice=voice_name, speed=1, split_pattern=r'\n+')
        
        full_audio = []
        for _, _, audio in generator:
            full_audio.append(audio)
        
        # Combine all parts into one array
        audio_combined = np.concatenate(full_audio)

        # 3. Convert to MP3 for LiveKit
        # First write to a WAV buffer
        wav_io = io.BytesIO()
        sf.write(wav_io, audio_combined, 24000, format='WAV')
        wav_io.seek(0)
        
        # Convert WAV to MP3 using pydub
        audio_segment = AudioSegment.from_wav(wav_io)
        mp3_io = io.BytesIO()
        audio_segment.export(mp3_io, format="mp3", bitrate="128k")
        mp3_io.seek(0)

        return send_file(mp3_io, mimetype="audio/mpeg")

    except Exception as e:
        print(f"❌ Kokoro Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print(f"🚀 Kokoro TTS Server starting on port 5001 (Device: {device})")
    app.run(host='0.0.0.0', port=5001, debug=False)
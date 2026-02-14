from flask import Flask, request, jsonify
import whisper
import torch
import numpy as np
from io import BytesIO
import base64
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load Whisper model (runs locally)
print("Loading Whisper model...")
model = whisper.load_model("large")  # or "base", "small", "medium", "large"
print("✅ Whisper model loaded!")

@app.route('/v1/audio/transcriptions', methods=['POST'])
def transcribe():
    try:
        # Parse request
        audio_file = request.files.get('file')
        
        if audio_file:
            # Save to temp file
            audio_path = "/tmp/temp_audio.wav"
            audio_file.save(audio_path)
            
            # Transcribe
            result = model.transcribe(audio_path, fp16=False)
            
            return jsonify({
                "text": result["text"],
                "language": result["language"]
            })
        
        # Alternative: Base64 encoded audio
        if request.json and 'file' in request.json:
            audio_b64 = request.json['file'].split(',')[1]
            audio_bytes = base64.b64decode(audio_b64)
            
            # Convert to numpy array
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Transcribe
            result = model.transcribe(audio_np, fp16=False)
            
            return jsonify({
                "text": result["text"],
                "language": result["language"]
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
    return jsonify({"error": "No audio provided"}), 400

@app.route('/v1/models', methods=['GET'])
def list_models():
    return jsonify({
        "data": [{
            "id": "whisper-1",
            "object": "model",
            "created": 1677678600,
            "owned_by": "openai"
        }]
    })

if __name__ == '__main__':
    print("🚀 Starting local Whisper server on http://localhost:9000")
    print("📝 Endpoint: POST http://localhost:9000/v1/audio/transcriptions")
    app.run(host='0.0.0.0', port=9000, debug=False)
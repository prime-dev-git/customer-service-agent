# Customer Service AI Agent — Voice-Enabled RAG Assistant

A voice AI agent for customer service interactions. Features natural voice conversations with real-time interruption handling, RAG-powered responses grounded in company documentation, and persistent session management all running on your infrastructure with no external API dependencies.
 
*Full voice pipeline: Whisper STT → Ollama Agent w/ RAG → Kokoro TTS*

## ✨ Key Features

- **Natural Voice Conversations**  
  Full-duplex interaction with human-like interruption handling during both speech and thinking phases
  
- **Document-Grounded Responses**  
  RAG system retrieves answers exclusively from your company documentation (Word/PDF support)
  
- **Field-by-Field Form Processing**  
  Guides customers through multi-field forms (e.g., Visitor Access Requests) with validation at each step
  
- **Session Persistence**  
  Automatically saves/resumes conversations with full history across sessions
  
- **Local-First Architecture**  
  Zero external dependencies — all components run on your hardware:
  - Whisper (speech-to-text)
  - Kokoro (high-quality TTS)
  - Ollama + LangChain (LLM agent)
  - Chroma (vector database)

- **Smart Interruption Handling**  
  Ignores filler words ("okay", "yeah") while responding instantly to meaningful interruptions

## 🏗️ System Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    CUSTOMER INTERACTION                       │
└──────────────────────────┬────────────────────────────────────┘
                           │
        ┌──────────────────▼──────────────────┐
        │   VOICE ACTIVITY DETECTION (VAD)    │
        │   • Real-time silence detection     │
        │   • Significant audio filtering     │
        └──────────────────┬──────────────────┘
                           │
        ┌──────────────────▼──────────────────┐
        │   WHISPER STT SERVER (port 9000)    │
        │   • Local large-v2 transcription    │
        │   • Base64/WAV file support         │
        └──────────────────┬──────────────────┘
                           │
        ┌──────────────────▼──────────────────┐
        │   AGENT CORE (LangChain + Ollama)   │
        │   • RAG context retrieval           │
        │   • Field-by-field form handling    │
        │   • Session-aware conversation      │
        └──────────────────┬──────────────────┘
                           │
        ┌──────────────────▼──────────────────┐
        │   KOKORO TTS SERVER (port 5001)     │
        │   • GPU-accelerated speech synthesis│
        │   • Multiple voice options          │
        └──────────────────┬──────────────────┘
                           │
        ┌──────────────────▼──────────────────┐
        │        AUDIO OUTPUT (MP3)           │
        │   • ffplay playback                 │
        │   • Real-time interruption checks   │
        └─────────────────────────────────────┘
```

## ⚙️ Prerequisites

### Hardware
- **GPU strongly recommended** (NVIDIA with 8GB+ VRAM for Kokoro + Whisper)
- Minimum 16GB RAM (32GB recommended for smooth operation)
- Microphone and speakers/headphones

### Software
```bash
# Core dependencies
Python 3.10+ 
FFmpeg (for audio processing)
Ollama (https://ollama.com)
```

### Required Models
```bash
# Install via Ollama
ollama pull ministral-3:3b      # Main agent model
ollama pull nomic-embed-text    # Embedding model

# Whisper model (auto-downloaded on first run)
# Kokoro model (auto-downloaded on first run)
```

## 🚀 Installation

```bash
# 1. Clone repository
git clone https://github.com/your-username/customer-service-agent.git
cd customer-service-agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Prepare directories
mkdir -p audio_output chat_sessions agent/doc

# 5. Add your documentation
# Place company documents in ./agent/doc/
# Supported formats: .docx, .pdf, .txt
cp /path/to/your/forms.docx agent/doc/
```

## ⚙️ Configuration

Edit key parameters in `agent.py`:

```python
# agent/customer_service.py
company = "JQW Ltd."  # ← YOUR COMPANY NAME
enquiry = "Visitor Access Request form"  # ← YOUR USE CASE
purpose = "Request details needed field by field..."  # ← CONVERSATION FLOW RULES
```

### Voice Settings (`voice_assistant.py`)
```python
# TTS/STT endpoints (default ports)
WHISPER_URL = "http://localhost:9000/v1/audio/transcriptions"
KOKORO_URL = "http://localhost:5001/v1/audio/speech"

# Voice selection (Kokoro voices)
VOICE_OPTIONS = ["af_heart", "af_bella", "am_adam", "am_michael"]

# VAD sensitivity (adjust for your environment)
SILENCE_THRESHOLD = 0.02   # Lower = more sensitive
SILENCE_DURATION = 1.5     # Seconds before stopping recording
```

## ▶️ Usage

### Step 1: Start Required Services
```bash
# Terminal 1: Start Whisper STT server
python whisper_server.py

# Terminal 2: Start Kokoro TTS server
python kokoro_server.py
```

### Step 2: Launch Voice Agent
```bash
# Start NEW session
python agent.py

# Resume EXISTING session
python agent.py <session-id>

# List available sessions
python agent.py --list
```

### During Conversation
- **Speak naturally** — system detects speech boundaries automatically
- **Interrupt anytime** — say "wait" or "actually" to stop AI mid-sentence
- **Exit gracefully** — say "goodbye" or press `Ctrl+C` to save session

## 💼 Sample Interaction Flow

```
🤖 AI: "Hello! I'm Alex from JQW Ltd. How can I help you today?"
👤 You: "I need to request visitor access for my client"
🤖 AI: "I'd be happy to help with that visitor access request. 
         First, may I have the visitor's full name?"
👤 You: "Sarah Johnson"
🤖 AI: "Got it — Sarah Johnson. Next, what's the purpose of their visit?"
👤 You: "Client meeting in conference room B"
🤖 AI: "Purpose noted: client meeting in conference room B. 
         What date will they be visiting?"
...
```

## 🛠️ Customization Guide

### Adding New Documentation
1. Place documents in `agent/doc/`
2. Restart agent — documents auto-ingest on startup
3. Agent will ONLY answer using these documents

### Changing Conversation Behavior
Modify `system_prompt` in `CustomerService.__init__()`:
```python
self.system_prompt = f"""You are a customer service representative for {company}.

YOUR PURPOSE:
- Handle inquiries about {enquiry}
- {purpose}  # ← Customize field collection logic here
- Be professional and friendly

CRITICAL RULES:
1. ALWAYS use rag_context tool before answering
2. NEVER invent information not in documents
3. For forms: collect ONE field at a time, confirm, then proceed
"""
```

### Voice Selection
Change voice in `speak_text()` call:
```python
# Available Kokoro voices:
# af_heart (female, warm)   | af_bella (female, professional)
# am_adam (male, calm)      | am_michael (male, energetic)
self.speak_text(response, voice="af_bella")
```

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce Whisper model size (`whisper.load_model("small")`) |
| No audio playback | Install FFmpeg: `sudo apt install ffmpeg` (Linux) / `brew install ffmpeg` (Mac) |
| Agent ignores documents | Verify documents exist in `agent/doc/` and restart agent |
| False interruption triggers | Increase `SILENCE_THRESHOLD` (e.g., `0.03` → `0.05`) |
| Slow response times | Use smaller LLM (`ministral:3b` → `phi3:mini`) |

## 🔒 Privacy & Security

- ✅ All data stays on your machines — no cloud APIs
- ✅ Audio never leaves your device
- ✅ Session files stored locally in `chat_sessions/`
- ✅ No telemetry or external connections beyond localhost services

## 📜 License

This project is for internal business use. Components use their respective licenses:
- Kokoro TTS: Research use only
- Whisper: MIT License
- Ollama models: Model-specific licenses (check ollama.com/library)



> 💡 **Pro Tip**: For production deployment, wrap servers in systemd services and add health checks. Always test with actual customer queries before deployment!

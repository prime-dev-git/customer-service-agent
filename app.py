import hashlib
import requests
import os
import torch
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import queue
from langchain_ollama import ChatOllama
from collections import deque
import time
import json
from datetime import datetime
import uuid
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from agent import customer_service
from agent.tool import rag
from agent.tool.rag import rag_context
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# --- CONFIGURATION ---
WHISPER_URL = "http://localhost:9000/v1/audio/transcriptions" 
KOKORO_URL = "http://localhost:5001/v1/audio/speech"
OLLAMA_MODEL = "ministral-3:3b"
EMBEDDINGS = "nomic-embed-text:latest"

# Voice Activity Detection settings
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.02  # Adjust based on your microphone
SILENCE_DURATION = 1.5  # Seconds of silence before stopping
CHUNK_DURATION = 0.1  # Process audio in 100ms chunks
INTERRUPT_CHECK_DURATION = 1.0  # Capture 1 second to check for interruption

# Chat history settings
HISTORY_DIR = "chat_sessions"
MAX_HISTORY_MESSAGES = 20  # Keep last 20 messages in context

# Words that should NOT interrupt when said alone
IGNORE_WORDS = [
    "okay", "ok", "nice", "yeah", "yes", "yep", "yup", "sure",
    "alright", "right", "mhm", "uh-huh", "got it", "cool", "great",
    "mm-hmm", "uh huh"
]



class VoiceAssistant:
    def __init__(self, session_id=None):
        self.is_speaking = False
        self.should_stop_speaking = False
        self.interrupt_monitoring = False
        self.interrupt_buffer = []
        self.agent = customer_service.CustomerSevice()
        self.interrupt_lock = threading.Lock()
        
        self.llm = ChatOllama(model=OLLAMA_MODEL)
        self.embeddings = OllamaEmbeddings(model=EMBEDDINGS)
        self.vector_store = Chroma(
            collection_name="service",
            embedding_function=self.embeddings,
            persist_directory="./agent",
        )
        
        # Set the vector store for the tool
        rag.set_vector_store(self.vector_store)
        
        # Now create the agent with the tool
        self.tools = [rag_context]
        
        self.llm_agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.agent.system_prompt
        )
        
        # Session management
        self.session_id = session_id or str(uuid.uuid4())
        self.chat_history = []
        self.conversation_started = False  # Track if we've greeted the user
        
        # Create history directory if it doesn't exist
        os.makedirs(HISTORY_DIR, exist_ok=True)
        
        # Load existing session or create new one
        self.session_file = os.path.join(HISTORY_DIR, f"{self.session_id}.json")
        self.load_session()
    
        
    def load_session(self):
        """Load chat history from session file"""
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file, 'r') as f:
                    data = json.load(f)
                    self.chat_history = data.get('messages', [])
                    self.conversation_started = len(self.chat_history) > 0
                    print(f"📂 Loaded session: {self.session_id}")
                    print(f"📜 Found {len(self.chat_history)} previous messages")
            except Exception as e:
                print(f"⚠️ Could not load session: {e}")
                self.chat_history = []
                self.conversation_started = False
        else:
            print(f"✨ New session created: {self.session_id}")
            self.conversation_started = False
            self.save_session()
    
    def save_session(self):
        """Save chat history to session file"""
        try:
            session_data = {
                'session_id': self.session_id,
                'created_at': datetime.now().isoformat(),
                'messages': self.chat_history
            }
            with open(self.session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save session: {e}")
    
    def add_to_history(self, role, content):
        """Add message to chat history"""
        if content is None:
            print(f"⚠️ Skipping None content for role: {role}")
            return
            
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        self.chat_history.append(message)
        self.save_session()
    
    def get_context_messages(self):
        """Get recent messages for LLM context in LangGraph format"""
        # Keep only the last MAX_HISTORY_MESSAGES
        recent_messages = self.chat_history[-MAX_HISTORY_MESSAGES:]
        
        # Convert to LangGraph message format
        messages = []
        for msg in recent_messages:
            if msg.get('content') is None:
                continue
                
            if msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                messages.append(AIMessage(content=msg['content']))
        
        return messages
    
    def get_initial_greeting(self):
        """Get initial greeting from agent"""
        try:
            # Create a specific prompt for greeting based on agent's purpose
            greeting_prompt = HumanMessage(content="[SYSTEM: This is the start of a new call. Please greet the caller according to your role and purpose.]")
            
            inputs = {"messages": [greeting_prompt]}
            result = self.llm_agent.invoke(inputs)
            
            if result and "messages" in result and len(result["messages"]) > 0:
                return result["messages"][-1].content
            else:
                # Fallback greeting
                return "Hello! How can I help you today?"
                
        except Exception as e:
            print(f"❌ Greeting Error: {e}")
            return "Hello! How can I help you today?"
    
    def get_agent_response(self, user_input):
        """Get response from the agent"""
        try:
            context_messages = self.get_context_messages()
            
            # Add user message
            context_messages.append(HumanMessage(content=user_input))
            
            # Invoke the agent
            inputs = {"messages": context_messages}
            result = self.llm_agent.invoke(inputs)
            
            # Extract the response
            if result and "messages" in result and len(result["messages"]) > 0:
                return result["messages"][-1].content
            else:
                return "I apologize, I couldn't generate a response. Could you please repeat that?"
                
        except Exception as e:
            print(f"❌ Agent Error: {e}")
            import traceback
            traceback.print_exc()
            return "I encountered an error. Please try again."
    
    def is_silence(self, audio_chunk):
        """Check if audio chunk is below silence threshold"""
        return np.abs(audio_chunk).mean() < SILENCE_THRESHOLD
    
    def should_ignore_interruption(self, text):
        """Check if the interruption text should be ignored"""
        if not text:
            return True
        
        # Clean and normalize the text
        cleaned = text.lower().strip().rstrip('.,!?')
        
        # Check if it's just a single acknowledgment word
        if cleaned in IGNORE_WORDS:
            return True
        
        # Check if it's multiple words from ignore list (e.g., "yeah okay")
        words = cleaned.split()
        if len(words) > 0 and all(word in IGNORE_WORDS for word in words):
            return True
        
        return False
    
    def quick_transcribe(self, audio_data):
        """Quick transcription for interruption checking"""
        temp_file = "temp_interrupt.wav"
        sf.write(temp_file, audio_data, SAMPLE_RATE)
        
        try:
            with open(temp_file, "rb") as f:
                files = {"file": (temp_file, f, "audio/wav")}
                response = requests.post(WHISPER_URL, files=files, timeout=3)
            
            if response.status_code == 200:
                return response.json().get("text", "").strip()
        except Exception as e:
            print(f"⚠️ Quick transcribe error: {e}")
        
        return ""
    
    def monitor_interruptions(self):
        """Monitor for interruptions while AI is speaking"""
        def callback(indata, frames, time_info, status):
            if self.interrupt_monitoring:
                with self.interrupt_lock:
                    self.interrupt_buffer.append(indata.copy())
        
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, 
                          callback=callback, blocksize=int(SAMPLE_RATE * CHUNK_DURATION)):
            while self.is_speaking:
                time.sleep(0.1)
                
                # Check if we have enough audio to analyze
                with self.interrupt_lock:
                    if len(self.interrupt_buffer) >= int(INTERRUPT_CHECK_DURATION / CHUNK_DURATION):
                        # Check if there's actual speech
                        recent_chunks = self.interrupt_buffer[-int(0.5 / CHUNK_DURATION):]
                        has_speech = any(not self.is_silence(chunk) for chunk in recent_chunks)
                        
                        if has_speech:
                            # Transcribe the buffered audio
                            audio_data = np.concatenate(self.interrupt_buffer, axis=0)
                            text = self.quick_transcribe(audio_data)
                            
                            if text and not self.should_ignore_interruption(text):
                                print(f"\n🛑 Interrupted with: '{text}'")
                                self.should_stop_speaking = True
                                self.interrupt_buffer.clear()
                                break
                            elif text:
                                print(f"💬 Acknowledged: '{text}' (continuing...)")
                        
                        # Keep only recent chunks
                        self.interrupt_buffer = self.interrupt_buffer[-int(0.5 / CHUNK_DURATION):]
    
    def record_with_vad(self):
        """Record audio until silence is detected"""
        print("🎤 Listening... (speak now)")
        
        audio_buffer = []
        silence_chunks = 0
        max_silence_chunks = int(SILENCE_DURATION / CHUNK_DURATION)
        audio_queue = queue.Queue()
        self.has_significant_audio = False  # Track if we got meaningful audio
        
        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"Status: {status}")
            audio_queue.put(indata.copy())
        
        # Start recording stream
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, 
                          callback=audio_callback, blocksize=int(SAMPLE_RATE * CHUNK_DURATION)):
            recording_started = False
            
            while True:
                try:
                    chunk = audio_queue.get(timeout=0.1)
                    
                    if self.is_silence(chunk):
                        if recording_started:
                            silence_chunks += 1
                            audio_buffer.append(chunk)
                            
                            if silence_chunks >= max_silence_chunks:
                                print("✅ Recording stopped (silence detected)")
                                break
                    else:
                        # Sound detected - check if it's significant
                        chunk_level = np.abs(chunk).mean()
                        if chunk_level > SILENCE_THRESHOLD * 2:  # At least 2x the silence threshold
                            self.has_significant_audio = True
                        
                        if not recording_started:
                            print("🎙️ Recording started...")
                        recording_started = True
                        silence_chunks = 0
                        audio_buffer.append(chunk)
                        
                except queue.Empty:
                    continue
        
        if not audio_buffer:
            return None
            
        # Combine all chunks
        audio_data = np.concatenate(audio_buffer, axis=0)
        return audio_data
    
    def transcribe_audio(self, audio_data=None):
        """Transcribe audio using Whisper"""
        if audio_data is None:
            return ""
        
        # Save to temporary file
        temp_file = "input_mic.wav"
        sf.write(temp_file, audio_data, SAMPLE_RATE)
        
        print("☁️ Transcribing...")
        try:
            with open(temp_file, "rb") as f:
                files = {"file": (temp_file, f, "audio/wav")}
                response = requests.post(WHISPER_URL, files=files)
            
            if response.status_code != 200:
                print(f"❌ Whisper Error ({response.status_code}): {response.text}")
                return ""
            
            text = response.json().get("text", "").strip()
            return text
        except Exception as e:
            print(f"❌ Transcription Failed: {e}")
            return ""
    
    def monitor_thinking_interruption(self):
        """Monitor for interruptions while AI is thinking"""
        audio_queue = queue.Queue()
        
        def audio_callback(indata, frames, time_info, status):
            audio_queue.put(indata.copy())
        
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, 
                          callback=audio_callback, blocksize=int(SAMPLE_RATE * CHUNK_DURATION)):
            
            audio_buffer = []
            while not self.should_stop_speaking:  # Reusing this flag for thinking interruption
                try:
                    chunk = audio_queue.get(timeout=0.1)
                    
                    if not self.is_silence(chunk):
                        audio_buffer.append(chunk)
                        
                        # Check if we have enough audio to transcribe
                        if len(audio_buffer) >= int(0.8 / CHUNK_DURATION):
                            audio_data = np.concatenate(audio_buffer, axis=0)
                            text = self.quick_transcribe(audio_data)
                            
                            if text and not self.should_ignore_interruption(text):
                                print(f"\n🛑 Interrupted thinking with: '{text}'")
                                return text  # Return the interruption text
                            elif text:
                                print(f"💬 Acknowledged: '{text}' (continuing...)")
                                audio_buffer = []
                    else:
                        # Reset buffer on silence
                        if len(audio_buffer) > 0:
                            audio_buffer = []
                            
                except queue.Empty:
                    continue
        
        return None
    
    def speak_text(self, text, voice="af_heart"):
        """Generate and play TTS with interruption support"""
        if text is None or text.strip() == "":
            print("⚠️ Skipping empty text")
            return
            
        print(f"\n🔊 AI: {text}\n")
        
        payload = {"input": text, "voice": voice}
        try:
            response = requests.post(KOKORO_URL, json=payload)
            if response.status_code == 200:
                with open("./audio_ouput/output.mp3", "wb") as f:
                    f.write(response.content)
                
                self.is_speaking = True
                self.should_stop_speaking = False
                self.interrupt_monitoring = True
                
                with self.interrupt_lock:
                    self.interrupt_buffer.clear()
                
                # Start interrupt monitoring thread
                monitor_thread = threading.Thread(target=self.monitor_interruptions, daemon=True)
                monitor_thread.start()
                
                # Play audio in a separate thread
                def play_audio():
                    import subprocess
                    process = subprocess.Popen(
                        ["ffplay", "-nodisp", "-autoexit", "./audio_ouput/output.mp3"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    
                    # Monitor for interruption
                    while process.poll() is None:
                        if self.should_stop_speaking:
                            process.kill()
                            print("🔇 Speech stopped")
                            break
                        time.sleep(0.05)
                    
                    self.is_speaking = False
                    self.interrupt_monitoring = False
                
                play_thread = threading.Thread(target=play_audio)
                play_thread.start()
                play_thread.join()
                
            else:
                print(f"❌ TTS Error: {response.text}")
                self.is_speaking = False
                self.interrupt_monitoring = False
        except Exception as e:
            print(f"❌ Speak failed: {e}")
            self.is_speaking = False
            self.interrupt_monitoring = False
    
    def run(self):
        """Main conversation loop"""
        print("=" * 60)
        print("👋 Voice Assistant Started!")
        print(f"🆔 Session ID: {self.session_id}")
        print(f"⚙️  Silence threshold: {SILENCE_THRESHOLD}, Duration: {SILENCE_DURATION}s")
        print(f"🚫 Ignore words: {', '.join(IGNORE_WORDS[:5])}... (and more)")
        print("=" * 60)
        print()
        
        self.upsert_documents()
        
        # Start conversation with greeting if new session
        if not self.conversation_started:
            print("🤔 Preparing greeting...")
            greeting = self.get_initial_greeting()
            if greeting:
                self.add_to_history('assistant', greeting)
                self.speak_text(greeting)
                self.conversation_started = True
       
        while True:
            try:
                # Record with VAD
                audio_data = self.record_with_vad()
                
                if audio_data is None:
                    continue
                
                # Check if audio was too quiet
                if not self.has_significant_audio:
                    quiet_message = "I can't hear you properly. Could you please speak louder or move closer to the microphone?"
                    print(f"🔇 {quiet_message}")
                    self.speak_text(quiet_message)
                    continue
                
                # Transcribe
                user_text = self.transcribe_audio(audio_data)
                
                if not user_text:
                    retry_message = "I couldn't understand that. Could you please repeat that?"
                    print(f"🤷 {retry_message}")
                    self.speak_text(retry_message)
                    continue
                
                print(f"👤 You: {user_text}")
                
                # Add to history
                self.add_to_history('user', user_text)
                
                # Check for exit commands
                if user_text.lower() in ["exit", "quit", "goodbye", "stop", "bye"]:
                    response_text = "Goodbye! Have a great day! Feel free to call again if you need any assistance."
                    self.add_to_history('assistant', response_text)
                    self.speak_text(response_text)
                    print(f"\n💾 Session saved: {self.session_file}")
                    break
                
                # Get agent response
                print("🤔 Thinking...")
                
                # Start monitoring for interruptions during thinking
                self.should_stop_speaking = False
                interruption_text = None
                
                def get_llm_response():
                    nonlocal response_text
                    response_text = self.get_agent_response(user_text)
                
                response_text = None
                llm_thread = threading.Thread(target=get_llm_response)
                llm_thread.start()
                
                # Monitor for interruption while thinking
                self._interrupt_text = None
                monitor_thread = threading.Thread(
                    target=lambda: setattr(self, '_interrupt_text', self.monitor_thinking_interruption()), 
                    daemon=True
                )
                monitor_thread.start()
                
                # Wait for LLM to finish
                llm_thread.join()
                
                # Stop monitoring
                self.should_stop_speaking = True
                monitor_thread.join(timeout=1)
                
                # Check if interrupted
                interruption_text = getattr(self, '_interrupt_text', None)
                
                if interruption_text:
                    print("🔄 User interrupted with new input, processing...")
                    # Add the interruption as a new user message
                    self.add_to_history('user', interruption_text)
                    
                    # Process the interruption as new input
                    print(f"👤 You (interrupt): {interruption_text}")
                    
                    # Check for exit in interruption
                    if interruption_text.lower() in ["exit", "quit", "goodbye", "stop", "bye"]:
                        response_text = "Goodbye! Have a great day! Feel free to call again if you need any assistance."
                        self.add_to_history('assistant', response_text)
                        self.speak_text(response_text)
                        print(f"\n💾 Session saved: {self.session_file}")
                        break
                    
                    # Get new response for interruption
                    print("🤔 Thinking (processing interruption)...")
                    response_text = self.get_agent_response(interruption_text)
                
                # Add response to history and speak
                if response_text:
                    self.add_to_history('assistant', response_text)
                    self.speak_text(response_text)
                else:
                    print("⚠️ No response generated")
                
            except KeyboardInterrupt:
                print("\n\n👋 Shutting down...")
                print(f"💾 Session saved: {self.session_file}")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                continue

    def upsert_documents(self):
        """
        Add documents to a LangChain vector store in an idempotent way.
        Same content + same source => same ID => no duplicates.
        Changed content => new ID => updated embedding.
        """
        def make_id(doc):
            source = doc.metadata.get("source", "")
            content = doc.page_content
            base = f"{source}::{content}".encode("utf-8")
            return hashlib.sha256(base).hexdigest()

        ids = [make_id(d) for d in self.agent.docs]

        # Most vector stores (Chroma, Qdrant, Pinecone, Weaviate) will upsert by ID
        self.vector_store.add_documents(documents=self.agent.docs, ids=ids)
        print(f"✅ Upserted {len(ids)} documents to vector store")

        return ids

def list_sessions():
    """List all available sessions"""
    if not os.path.exists(HISTORY_DIR):
        return []
    
    sessions = []
    for filename in os.listdir(HISTORY_DIR):
        if filename.endswith('.json'):
            filepath = os.path.join(HISTORY_DIR, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    sessions.append({
                        'id': data['session_id'],
                        'created_at': data.get('created_at', 'Unknown'),
                        'message_count': len(data.get('messages', []))
                    })
            except:
                pass
    
    return sessions

if __name__ == "__main__":
    import sys
    
    # Check if session ID provided as argument
    if len(sys.argv) > 1:
        if sys.argv[1] == '--list':
            print("\n📋 Available Sessions:")
            print("=" * 60)
            sessions = list_sessions()
            if sessions:
                for session in sessions:
                    print(f"🆔 {session['id']}")
                    print(f"   Created: {session['created_at']}")
                    print(f"   Messages: {session['message_count']}")
                    print()
            else:
                print("No sessions found.")
            print("=" * 60)
            sys.exit(0)
        else:
            session_id = sys.argv[1]
            assistant = VoiceAssistant(session_id=session_id)
    else:
        # Create new session
        assistant = VoiceAssistant()
    
    assistant.run()
import numpy as np
import librosa
import parselmouth 
import whisper
from fastapi import FastAPI, UploadFile, File, Response, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import shutil
import subprocess
import json
from openai import OpenAI
from io import BytesIO
import uuid

app = FastAPI()

# --- CORS CONFIGURATION (Crucial for Localhost) ---
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*" # For development only, allows any origin
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client (if API key is available)
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = None
if openai_api_key:
    openai_client = OpenAI(api_key=openai_api_key)
    print("OpenAI client initialized successfully!")
else:
    print("Warning: OPENAI_API_KEY not set. Vocabulary and grammar analysis will use default values.")

# Load Models
print("Loading Whisper Model...")
try:
    model = whisper.load_model("base")
    print("Whisper Model loaded successfully!")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    print("This might be a network issue. The model should be pre-downloaded during Docker build.")
    raise

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the backend is running and ready"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "message": "Backend is ready to process audio"
    } 

# --- Response Models ---
class MetricDetail(BaseModel):
    score: float
    feedback: str
    # Flexible dict to allow extra fields like 'wpm' or 'unique_words'
    # In a strict schema, you would define these explicitly.

class AnalysisResult(BaseModel):
    overall_band: float
    transcript: str
    fluency: dict
    pronunciation: dict
    vocabulary: dict
    grammar: dict

# --- Conversation Models ---
class ConversationMessage(BaseModel):
    role: str  # "assistant" or "user"
    content: str
    audio_url: Optional[str] = None  # URL to audio for assistant messages

class ConversationRequest(BaseModel):
    topic: Optional[str] = None
    conversation_id: Optional[str] = None
    conversation_history: List[ConversationMessage] = []
    user_response: Optional[str] = None  # Transcript of user's latest response

class ConversationResponse(BaseModel):
    message: str
    audio_url: str
    conversation_id: Optional[str] = None

class TranscribeRequest(BaseModel):
    conversation_id: Optional[str] = None

# In-memory conversation storage (in production, use a database)
conversations: Dict[str, List[ConversationMessage]] = {}
IELTS_TOPICS = [
    "Music", "Sports", "Travel", "Food", "Hobbies", "Work", "Education",
    "Family", "Technology", "Art", "Books", "Nature", "Shopping", "Health"
]

# Ensure temp directory exists for audio files
TEMP_AUDIO_DIR = "/tmp"
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

def analyze_intonation(audio_path):
    sound = parselmouth.Sound(audio_path)
    pitch = sound.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values != 0]
    
    if len(pitch_values) == 0:
        return 0.0, "Monotone"
    
    pitch_std = np.std(pitch_values)
    
    feedback = "Good intonation range."
    if pitch_std < 20:
        feedback = "Speech is somewhat flat/monotone."
    elif pitch_std > 50:
        feedback = "Very expressive intonation."
        
    return float(pitch_std), feedback

def analyze_fluency(audio_path, transcript_word_count):
    y, sr = librosa.load(audio_path)
    total_duration = librosa.get_duration(y=y, sr=sr)
    
    if total_duration == 0:
        return {"wpm": 0, "pauses": 0, "score": 1.0, "feedback": "Audio too short."}

    wpm = (transcript_word_count / total_duration) * 60
    
    non_silent_intervals = librosa.effects.split(y, top_db=25)
    long_pauses = 0
    for i in range(len(non_silent_intervals) - 1):
        pause_len = (non_silent_intervals[i+1][0] - non_silent_intervals[i][1]) / sr
        if pause_len > 0.5:
            long_pauses += 1
            
    score = 6.0
    feedback = "Average fluency."
    
    if wpm > 120 and long_pauses < 3:
        score = 7.5
        feedback = "Excellent flow with minimal hesitation."
    elif wpm < 80 or long_pauses > 6:
        score = 5.0
        feedback = "Frequent pauses disrupted the flow of speech."
        
    return {
        "score": score,
        "wpm": round(wpm, 1),
        "pauses": long_pauses,
        "filled_pauses": 0, # Would need NLP to detect 'um'/'uh'
        "feedback": feedback
    }

def convert_to_wav(input_path, output_path):
    """Convert audio file to WAV format using ffmpeg"""
    try:
        subprocess.run(
            ["ffmpeg", "-i", input_path, "-y", "-acodec", "pcm_s16le", "-ar", "44100", output_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error converting audio: {e}")
        return False

async def analyze_vocabulary_with_openai(transcript: str) -> dict:
    """Analyze lexical resource using OpenAI API"""
    if not openai_client:
        # Fallback to basic analysis if OpenAI is not available
        word_count = len(transcript.split())
        unique_words = len(set(word.lower() for word in transcript.split()))
        return {
            "score": 6.0,
            "unique_words": unique_words,
            "complexity": "Medium",
            "feedback": "OpenAI API key not configured. Using basic vocabulary analysis."
        }
    
    prompt = f"""Analyze the lexical resource (vocabulary) of the following IELTS speaking transcript. 
Evaluate it according to IELTS band descriptors for Lexical Resource:
- Band 9: Uses vocabulary with full flexibility and precision
- Band 7-8: Uses vocabulary resource flexibly to discuss a variety of topics, uses some less common vocabulary
- Band 5-6: Has enough vocabulary to discuss topics at length, uses some less common vocabulary with awareness of style
- Band 4: Can talk about familiar topics but vocabulary is limited
- Band 2-3: Only basic vocabulary is used

Transcript: "{transcript}"

Provide a JSON response with:
1. "score": A band score from 0.0 to 9.0
2. "unique_words": Count of unique words used
3. "complexity": One of "Basic", "Intermediate", "Advanced", or "Sophisticated"
4. "feedback": Detailed feedback explaining the score, mentioning specific vocabulary choices, use of less common words, collocations, and any areas for improvement

Respond ONLY with valid JSON, no additional text."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an IELTS examiner specializing in lexical resource evaluation. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return {
            "score": float(result.get("score", 6.0)),
            "unique_words": int(result.get("unique_words", len(set(word.lower() for word in transcript.split())))),
            "complexity": result.get("complexity", "Medium"),
            "feedback": result.get("feedback", "Vocabulary analysis completed.")
        }
    except Exception as e:
        print(f"Error in OpenAI vocabulary analysis: {e}")
        # Fallback
        word_count = len(transcript.split())
        unique_words = len(set(word.lower() for word in transcript.split()))
        return {
            "score": 6.0,
            "unique_words": unique_words,
            "complexity": "Medium",
            "feedback": f"Error analyzing vocabulary: {str(e)}"
        }

async def analyze_grammar_with_openai(transcript: str) -> dict:
    """Analyze grammar range and accuracy using OpenAI API"""
    if not openai_client:
        # Fallback to basic analysis if OpenAI is not available
        return {
            "score": 6.0,
            "errors": 0,
            "complexity": "Simple",
            "feedback": "OpenAI API key not configured. Using basic grammar analysis."
        }
    
    prompt = f"""Analyze the grammatical range and accuracy of the following IELTS speaking transcript.
Evaluate it according to IELTS band descriptors for Grammatical Range and Accuracy:
- Band 9: Uses a full range of structures naturally and appropriately, produces consistently accurate structures
- Band 7-8: Uses a range of complex structures with some flexibility, produces frequent error-free sentences
- Band 5-6: Uses a mix of simple and complex sentence forms, makes some errors in grammar and punctuation
- Band 4: Uses only basic sentence structures, makes frequent errors
- Band 2-3: Only basic sentence structures are attempted, errors predominate

Transcript: "{transcript}"

Provide a JSON response with:
1. "score": A band score from 0.0 to 9.0
2. "errors": Count of grammatical errors found
3. "complexity": One of "Simple", "Moderate", "Complex", or "Sophisticated"
4. "feedback": Detailed feedback explaining the score, mentioning specific grammatical structures used (tenses, conditionals, passive voice, etc.), errors found, and suggestions for improvement

Respond ONLY with valid JSON, no additional text."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an IELTS examiner specializing in grammatical range and accuracy evaluation. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return {
            "score": float(result.get("score", 6.0)),
            "errors": int(result.get("errors", 0)),
            "complexity": result.get("complexity", "Simple"),
            "feedback": result.get("feedback", "Grammar analysis completed.")
        }
    except Exception as e:
        print(f"Error in OpenAI grammar analysis: {e}")
        # Fallback
        return {
            "score": 6.0,
            "errors": 0,
            "complexity": "Simple",
            "feedback": f"Error analyzing grammar: {str(e)}"
        }

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_audio(file: UploadFile = File(...)):
    # Save Upload
    temp_filename = f"temp_{file.filename}"
    temp_wav_filename = "temp_audio.wav"
    
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Convert to WAV if needed (for parselmouth compatibility)
        if not temp_filename.lower().endswith('.wav'):
            if not convert_to_wav(temp_filename, temp_wav_filename):
                raise Exception("Failed to convert audio to WAV format")
            audio_path_for_parselmouth = temp_wav_filename
        else:
            audio_path_for_parselmouth = temp_filename
        
        # Use original file for Whisper (it supports many formats including WebM)
        # 1. Transcribe
        result = model.transcribe(temp_filename)
        transcript = result["text"]
        word_count = len(transcript.split())

        # 2. Analyze Signal
        # Use WAV for parselmouth, original for librosa (librosa can handle WebM)
        fluency_stats = analyze_fluency(temp_filename, word_count)
        pitch_std, pitch_feedback = analyze_intonation(audio_path_for_parselmouth)

        # 3. Analyze with OpenAI (Vocabulary and Grammar)
        vocabulary_stats = await analyze_vocabulary_with_openai(transcript)
        grammar_stats = await analyze_grammar_with_openai(transcript)

        # 4. Calculate pronunciation score
        pronunciation_score = 6.0 + (min(pitch_std, 60) / 20.0) # Rough heuristic
        pronunciation_score = min(9.0, pronunciation_score)

        # 5. Calculate overall band score
        overall_band = round(
            (fluency_stats['score'] + pronunciation_score + vocabulary_stats['score'] + grammar_stats['score']) / 4, 
            1
        )

        return {
            "overall_band": overall_band,
            "transcript": transcript,
            "fluency": fluency_stats,
            "pronunciation": {
                "score": round(pronunciation_score, 1),
                "pitch_variance": round(pitch_std, 1),
                "stress_accuracy": "Evaluated",
                "feedback": pitch_feedback
            },
            "vocabulary": vocabulary_stats,
            "grammar": grammar_stats
        }
    finally:
        # Cleanup
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        if os.path.exists(temp_wav_filename):
            os.remove(temp_wav_filename)

# --- Conversation & TTS Endpoints ---

@app.post("/conversation/start")
async def start_conversation(request: ConversationRequest):
    """Start a new IELTS conversation on a given topic"""
    conversation_id = str(uuid.uuid4())
    topic = request.topic or "General"
    
    if not openai_client:
        return {
            "message": "Hello! Let's begin our IELTS speaking practice. What would you like to talk about?",
            "audio_url": None,
            "conversation_id": conversation_id
        }
    
    # Generate initial question based on topic
    system_prompt = f"""You are an IELTS examiner conducting Part 1 of the speaking test. 
    Start a natural, friendly conversation about {topic}. 
    Ask one clear, open-ended question that encourages the candidate to speak for 1-2 minutes.
    Keep your question concise (1-2 sentences). Be conversational and warm."""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Start a conversation about {topic}."}
            ],
            temperature=0.7,
            max_tokens=100
        )
        
        initial_message = response.choices[0].message.content.strip()
        
        # Generate TTS audio
        audio_response = openai_client.audio.speech.create(
            model="tts-1",
            voice="alloy",  # Options: alloy, echo, fable, onyx, nova, shimmer
            input=initial_message
        )
        
        # Save audio to temporary file
        audio_filename = f"tts_{conversation_id}.mp3"
        audio_path = f"/tmp/{audio_filename}"
        # stream_to_file expects a file path string, not a file object
        audio_response.stream_to_file(audio_path)
        
        # Store conversation
        conversations[conversation_id] = [
            ConversationMessage(role="assistant", content=initial_message, audio_url=f"/audio/{audio_filename}")
        ]
        
        return {
            "message": initial_message,
            "audio_url": f"/audio/{audio_filename}",
            "conversation_id": conversation_id
        }
    except Exception as e:
        print(f"Error starting conversation: {e}")
        fallback_message = f"Let's talk about {topic}. Can you tell me about your experience with it?"
        return {
            "message": fallback_message,
            "audio_url": None,
            "conversation_id": conversation_id
        }

@app.post("/conversation/respond")
async def respond_in_conversation(request: ConversationRequest):
    """Continue the conversation with AI response based on user's input"""
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    if not openai_client:
        return {
            "message": "That's interesting! Can you tell me more?",
            "audio_url": None,
            "conversation_id": conversation_id
        }
    
    # Get conversation history
    history = conversations.get(conversation_id, [])
    
    # Add user's latest response to history
    if request.user_response:
        history.append(ConversationMessage(role="user", content=request.user_response))
    
    # Generate AI response based on conversation context
    system_prompt = """You are an IELTS examiner conducting Part 1 of the speaking test.
    You are having a natural, friendly conversation with a candidate. 
    Based on what they said, ask a follow-up question or make a comment that encourages them to continue speaking.
    Keep your response conversational, brief (1-2 sentences), and relevant to what they mentioned.
    Your goal is to keep the conversation flowing naturally while assessing their speaking ability."""
    
    try:
        messages_for_ai = [{"role": "system", "content": system_prompt}]
        for msg in history[-6:]:  # Keep last 6 messages for context
            messages_for_ai.append({"role": msg.role, "content": msg.content})
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_for_ai,
            temperature=0.7,
            max_tokens=100
        )
        
        ai_message = response.choices[0].message.content.strip()
        
        # Generate TTS audio
        audio_response = openai_client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=ai_message
        )
        
        # Save audio
        audio_filename = f"tts_{conversation_id}_{len(history)}.mp3"
        audio_path = f"/tmp/{audio_filename}"
        # stream_to_file expects a file path string, not a file object
        audio_response.stream_to_file(audio_path)
        
        # Update conversation history
        ai_msg_obj = ConversationMessage(
            role="assistant", 
            content=ai_message, 
            audio_url=f"/audio/{audio_filename}"
        )
        history.append(ai_msg_obj)
        conversations[conversation_id] = history
        
        return {
            "message": ai_message,
            "audio_url": f"/audio/{audio_filename}",
            "conversation_id": conversation_id
        }
    except Exception as e:
        print(f"Error responding in conversation: {e}")
        fallback_message = "That's interesting! Can you elaborate on that?"
        return {
            "message": fallback_message,
            "audio_url": None,
            "conversation_id": conversation_id
        }

@app.post("/conversation/transcribe")
async def transcribe_user_audio(file: UploadFile = File(...), conversation_id: Optional[str] = Form(None)):
    """Transcribe user's audio for conversation flow"""
    temp_filename = f"temp_transcribe_{file.filename}"
    
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = model.transcribe(temp_filename)
        transcript = result["text"].strip()
        
        # Update conversation history with user's transcript
        if conversation_id and conversation_id in conversations:
            conversations[conversation_id].append(
                ConversationMessage(role="user", content=transcript)
            )
        
        return {
            "transcript": transcript,
            "conversation_id": conversation_id
        }
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve generated TTS audio files"""
    audio_path = f"/tmp/{filename}"
    if os.path.exists(audio_path):
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        return Response(content=audio_data, media_type="audio/mpeg")
    else:
        return Response(content="Audio not found", status_code=404)

@app.get("/conversation/topics")
async def get_topics():
    """Get list of available IELTS conversation topics"""
    return {"topics": IELTS_TOPICS}

@app.post("/conversation/analyze-turn")
async def analyze_conversation_turn(file: UploadFile = File(...), conversation_id: Optional[str] = Form(None)):
    """Analyze a single turn in the conversation (for real-time feedback)"""
    temp_filename = f"temp_{file.filename}"
    temp_wav_filename = "temp_audio.wav"
    
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Convert to WAV if needed
        if not temp_filename.lower().endswith('.wav'):
            if not convert_to_wav(temp_filename, temp_wav_filename):
                raise Exception("Failed to convert audio to WAV format")
            audio_path_for_parselmouth = temp_wav_filename
        else:
            audio_path_for_parselmouth = temp_filename
        
        # Transcribe
        result = model.transcribe(temp_filename)
        transcript = result["text"]
        word_count = len(transcript.split())

        # Quick analysis (less detailed than full analyze)
        fluency_stats = analyze_fluency(temp_filename, word_count)
        pitch_std, pitch_feedback = analyze_intonation(audio_path_for_parselmouth)
        pronunciation_score = 6.0 + (min(pitch_std, 60) / 20.0)
        pronunciation_score = min(9.0, pronunciation_score)

        return {
            "transcript": transcript,
            "fluency": {
                "wpm": fluency_stats.get("wpm", 0),
                "pauses": fluency_stats.get("pauses", 0)
            },
            "pronunciation_score": round(pronunciation_score, 1),
            "feedback": "Continue speaking naturally."
        }
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        if os.path.exists(temp_wav_filename):
            os.remove(temp_wav_filename)

# Run with: uvicorn audio_analysis_service:app --reload --port 8000
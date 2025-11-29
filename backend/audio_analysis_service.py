import numpy as np
import librosa
import parselmouth 
import whisper
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import subprocess
import json
from openai import OpenAI

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

# Run with: uvicorn audio_analysis_service:app --reload --port 8000
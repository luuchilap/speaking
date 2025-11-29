import { useState, useRef, useEffect } from 'react';
import { Mic, Square, RefreshCw, BarChart2, Activity, Type, AlertCircle, ChevronDown, ChevronUp } from 'lucide-react';

// --- Real Backend Connection ---
const getBackendUrl = () => {
  return import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
};

const checkBackendHealth = async (): Promise<boolean> => {
  try {
    const backendUrl = getBackendUrl();
    const response = await fetch(`${backendUrl}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000), // 5 second timeout
    });
    return response.ok;
  } catch (error) {
    console.error("Backend health check failed:", error);
    return false;
  }
};

const analyzeAudioWithBackend = async (audioBlob: Blob) => {
  const formData = new FormData();
  // 'file' must match the parameter name in the Python FastAPI function
  formData.append('file', audioBlob, 'recording.webm');

  try {
    const backendUrl = getBackendUrl();
    
    // Check if backend is available before sending the request
    const isHealthy = await checkBackendHealth();
    if (!isHealthy) {
      throw new Error("Backend is not responding. Please check if the backend service is running.");
    }

    const response = await fetch(`${backendUrl}/analyze`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    return data;
  } catch (error: any) {
    console.error("Backend connection failed:", error);
    const errorMessage = error.message || "Could not connect to the Python backend.";
    alert(`${errorMessage}\n\nMake sure the backend is running on ${getBackendUrl()}\n\nIf using Docker, check: docker-compose ps`);
    return null;
  }
};

// --- Components ---

const ScoreCard = ({ title, score, details, icon: Icon, color }: any) => {
  const [expanded, setExpanded] = useState(false);

  // Helper to safely display details excluding metadata like 'score' and 'feedback'
  const renderDetails = () => {
    return Object.entries(details).map(([key, value]) => {
      if (['feedback', 'score'].includes(key)) return null;
      return (
        <div key={key} className="bg-white p-3 rounded border border-slate-200">
          <span className="block text-xs uppercase tracking-wider text-slate-400 mb-1">{key.replace('_', ' ')}</span>
          <span className="font-medium text-slate-700">{String(value)}</span>
        </div>
      );
    });
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-100 overflow-hidden transition-all duration-300 hover:shadow-md">
      <div 
        className="p-5 flex items-center justify-between cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-4">
          <div className={`p-3 rounded-full ${color} text-white`}>
            <Icon size={24} />
          </div>
          <div>
            <h3 className="font-semibold text-slate-800">{title}</h3>
            <p className="text-sm text-slate-500">{details.feedback ? details.feedback.substring(0, 50) + "..." : "No feedback"}</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-2xl font-bold text-slate-800">{score ? score.toFixed(1) : "-"}</div>
          {expanded ? <ChevronUp size={20} className="text-slate-400" /> : <ChevronDown size={20} className="text-slate-400" />}
        </div>
      </div>
      
      {expanded && (
        <div className="px-5 pb-5 pt-0 bg-slate-50 border-t border-slate-100">
          <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
            {renderDetails()}
          </div>
          <div className="mt-4 p-3 bg-blue-50 text-blue-800 text-sm rounded border border-blue-100">
            <strong>Feedback:</strong> {details.feedback}
          </div>
        </div>
      )}
    </div>
  );
};

const AudioVisualizer = ({ isRecording }: { isRecording: boolean }) => {
  return (
    <div className="h-24 bg-slate-900 rounded-xl flex items-center justify-center gap-1 overflow-hidden px-10">
      {[...Array(20)].map((_, i) => (
        <div 
          key={i}
          className={`w-2 bg-indigo-500 rounded-full transition-all duration-100 ${isRecording ? 'animate-pulse' : ''}`}
          style={{ 
            height: isRecording ? `${Math.random() * 100}%` : '4px',
            animationDelay: `${i * 0.05}s`
          }}
        />
      ))}
    </div>
  );
};

export default function IELTSApp() {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [timer, setTimer] = useState(0);
  const timerRef = useRef<any>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);

  // Timer Logic
  useEffect(() => {
    if (isRecording) {
      timerRef.current = setInterval(() => {
        setTimer((prev) => prev + 1);
      }, 1000);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [isRecording]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      chunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        processAudio(blob); // Auto-process on stop
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setResults(null);
      setTimer(0);
    } catch (err) {
      alert("Microphone access denied or not available.");
      console.error(err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }
  };

  const processAudio = async (blob: Blob) => {
    setIsProcessing(true);
    const data = await analyzeAudioWithBackend(blob);
    if (data) {
      setResults(data);
    }
    setIsProcessing(false);
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900 pb-20">
      <header className="bg-white border-b border-slate-200 py-4 sticky top-0 z-10">
        <div className="max-w-3xl mx-auto px-4 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center text-white font-bold">AI</div>
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-purple-600">
              IELTS Speaking Coach
            </h1>
          </div>
          <button className="text-sm font-medium text-slate-500 hover:text-indigo-600">History</button>
        </div>
      </header>

      <main className="max-w-3xl mx-auto px-4 mt-8">
        
        <div className="bg-indigo-600 rounded-2xl p-8 text-white shadow-lg mb-8">
          <span className="inline-block px-3 py-1 bg-indigo-500 rounded-full text-xs font-semibold tracking-wide mb-3">PART 1</span>
          <h2 className="text-2xl font-bold mb-2">Let's talk about Music.</h2>
          <p className="text-indigo-100 text-lg">"Do you prefer listening to live music or recorded music?"</p>
        </div>

        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 mb-8 text-center">
          <div className="mb-6">
            <AudioVisualizer isRecording={isRecording} />
          </div>
          
          <div className="text-4xl font-mono font-bold text-slate-700 mb-8">
            {formatTime(timer)}
          </div>

          <div className="flex justify-center gap-4">
            {!isRecording ? (
              <button 
                onClick={startRecording}
                disabled={isProcessing}
                className="group relative flex items-center justify-center w-20 h-20 bg-red-500 rounded-full shadow-lg hover:bg-red-600 transition-all hover:scale-105 active:scale-95 disabled:opacity-50"
              >
                <Mic size={32} className="text-white" />
                <span className="absolute -bottom-8 text-sm font-medium text-slate-500">Record</span>
              </button>
            ) : (
              <button 
                onClick={stopRecording}
                className="group relative flex items-center justify-center w-20 h-20 bg-slate-800 rounded-full shadow-lg hover:bg-slate-900 transition-all hover:scale-105 active:scale-95"
              >
                <Square size={28} className="text-white fill-current" />
                <span className="absolute -bottom-8 text-sm font-medium text-slate-500">Stop</span>
              </button>
            )}
          </div>
          
          {isProcessing && (
            <div className="mt-8 flex flex-col items-center animate-pulse">
              <div className="w-6 h-6 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin mb-2"></div>
              <p className="text-sm text-indigo-600 font-medium">Processing audio with AI (Whisper + Librosa)...</p>
            </div>
          )}
        </div>

        {results && !isProcessing && (
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-slate-800">Evaluation Report</h2>
              <div className="flex items-center gap-2 bg-indigo-50 px-4 py-2 rounded-lg border border-indigo-100">
                <span className="text-sm text-indigo-800 font-medium">Overall Band</span>
                <span className="text-2xl font-bold text-indigo-600">{results.overall_band}</span>
              </div>
            </div>

            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 mb-6">
              <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">Transcript</h3>
              <p className="text-lg leading-relaxed text-slate-700">
                "{results.transcript}"
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <ScoreCard 
                title="Fluency & Coherence"
                score={results.fluency.score}
                details={results.fluency}
                icon={Activity}
                color="bg-emerald-500"
              />
              <ScoreCard 
                title="Pronunciation"
                score={results.pronunciation.score}
                details={results.pronunciation}
                icon={BarChart2}
                color="bg-blue-500"
              />
              <ScoreCard 
                title="Lexical Resource"
                score={results.vocabulary.score}
                details={results.vocabulary}
                icon={Type}
                color="bg-purple-500"
              />
              <ScoreCard 
                title="Grammar Range"
                score={results.grammar.score}
                details={results.grammar}
                icon={AlertCircle}
                color="bg-amber-500"
              />
            </div>

            <button 
              onClick={() => { setResults(null); setTimer(0); }}
              className="mt-8 w-full py-4 bg-white border border-slate-200 text-slate-600 font-semibold rounded-xl hover:bg-slate-50 transition-colors flex items-center justify-center gap-2"
            >
              <RefreshCw size={20} />
              Try Another Question
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
import { useState, useRef, useEffect } from 'react';
import { Mic, Square, Clock, ArrowLeft, StopCircle, BarChart2, Activity, Type, AlertCircle, ChevronDown, ChevronUp, Loader2 } from 'lucide-react';

const getBackendUrl = () => {
  return import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
};

interface Part2ModeProps {
  onBack: () => void;
  onComplete: (topic: string) => void;
}

type Part2Phase = 'loading' | 'preparation' | 'speaking' | 'evaluating' | 'results';

export default function Part2Mode({ onBack, onComplete }: Part2ModeProps) {
  const [phase, setPhase] = useState<Part2Phase>('loading');
  const [taskCard, setTaskCard] = useState<any>(null);
  const [prepTime, setPrepTime] = useState(60);
  const [speakingTime, setSpeakingTime] = useState(120);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [evaluationResults, setEvaluationResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const prepTimerRef = useRef<any>(null);
  const speakingTimerRef = useRef<any>(null);
  const audioBlobRef = useRef<Blob | null>(null);

  const ScoreCard = ({ title, score, details, icon: Icon, color }: any) => {
    const [expanded, setExpanded] = useState(false);

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

  useEffect(() => {
    loadTaskCard();
  }, []);

  useEffect(() => {
    if (phase === 'preparation' && prepTime > 0) {
      prepTimerRef.current = setInterval(() => {
        setPrepTime(prev => {
          if (prev <= 1) {
            clearInterval(prepTimerRef.current);
            setPhase('speaking');
            startSpeaking();
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    } else {
      clearInterval(prepTimerRef.current);
    }

    return () => clearInterval(prepTimerRef.current);
  }, [phase, prepTime]);

  useEffect(() => {
    if (phase === 'speaking' && speakingTime > 0 && isRecording) {
      speakingTimerRef.current = setInterval(() => {
        setSpeakingTime(prev => {
          if (prev <= 1) {
            clearInterval(speakingTimerRef.current);
            stopRecording();
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    } else {
      clearInterval(speakingTimerRef.current);
    }

    return () => clearInterval(speakingTimerRef.current);
  }, [phase, speakingTime, isRecording]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  const loadTaskCard = async () => {
    try {
      const backendUrl = getBackendUrl();
      const response = await fetch(`${backendUrl}/ielts/part2/task-card`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to load task card');
      }

      const data = await response.json();
      setTaskCard(data);
      setPhase('preparation');
    } catch (error) {
      console.error('Error loading task card:', error);
      setError(`Failed to load task card: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  const startSpeaking = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      chunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        audioBlobRef.current = blob;
        await evaluatePart2(blob);
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (err) {
      alert('Microphone access denied or not available.');
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

  const evaluatePart2 = async (audioBlob: Blob) => {
    setIsProcessing(true);
    setPhase('evaluating');
    
    try {
      const backendUrl = getBackendUrl();
      const formData = new FormData();
      formData.append('file', audioBlob, 'part2_recording.webm');

      const response = await fetch(`${backendUrl}/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to evaluate');
      }

      const data = await response.json();
      setEvaluationResults(data);
      setPhase('results');
    } catch (error) {
      console.error('Error evaluating:', error);
      alert(`Failed to evaluate: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setPhase('speaking');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleContinue = () => {
    if (taskCard) {
      onComplete(taskCard.topic);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900 pb-20">
      <header className="bg-white border-b border-slate-200 py-4 sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-4 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center text-white font-bold">P2</div>
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-cyan-600">
              IELTS Speaking - Part 2
            </h1>
          </div>
          <button
            onClick={onBack}
            className="text-sm font-medium text-slate-500 hover:text-blue-600 flex items-center gap-1"
          >
            <ArrowLeft size={16} />
            Back
          </button>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 mt-8">
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-4 mb-6">
            <p className="text-red-800 text-sm">{error}</p>
            <button
              onClick={() => setError(null)}
              className="mt-2 text-sm text-red-600 hover:text-red-800 underline"
            >
              Dismiss
            </button>
          </div>
        )}

        {phase === 'loading' && (
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-12 text-center">
            <Loader2 className="animate-spin text-blue-600 mx-auto mb-4" size={32} />
            <p className="text-slate-600">Loading task card...</p>
          </div>
        )}

        {phase === 'preparation' && taskCard && (
          <div className="space-y-6">
            <div className="bg-blue-600 rounded-2xl p-6 text-white shadow-lg">
              <div className="flex items-center justify-between mb-4">
                <span className="text-sm font-semibold">Part 2: Long Turn</span>
                <div className="flex items-center gap-2 bg-blue-500 px-3 py-1 rounded-lg">
                  <Clock size={16} />
                  <span className="text-2xl font-mono font-bold">{formatTime(prepTime)}</span>
                </div>
              </div>
              <p className="text-sm opacity-90">You have 1 minute to prepare. Make notes if you wish.</p>
            </div>

            <div className="bg-white rounded-2xl shadow-sm border-2 border-blue-200 p-8">
              <h3 className="text-xl font-bold text-slate-800 mb-4">Describe {taskCard.topic}</h3>
              <p className="text-slate-700 mb-6">{taskCard.description}</p>
              
              <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
                <p className="text-sm font-semibold text-slate-600 mb-2">You should say:</p>
                <ul className="list-disc list-inside space-y-1 text-sm text-slate-700">
                  {taskCard.points?.map((point: string, index: number) => (
                    <li key={index}>{point}</li>
                  ))}
                </ul>
              </div>

              <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                <p className="text-sm text-yellow-800">
                  <strong>Note:</strong> You can make notes during the preparation time. 
                  When the timer reaches 0:00, you'll have 1-2 minutes to speak.
                </p>
              </div>
            </div>
          </div>
        )}

        {phase === 'speaking' && (
          <div className="space-y-6">
            <div className="bg-blue-600 rounded-2xl p-6 text-white shadow-lg">
              <div className="flex items-center justify-between mb-4">
                <span className="text-sm font-semibold">Part 2: Long Turn - Speaking</span>
                <div className="flex items-center gap-2 bg-blue-500 px-3 py-1 rounded-lg">
                  <Clock size={16} />
                  <span className="text-2xl font-mono font-bold">{formatTime(speakingTime)}</span>
                </div>
              </div>
              <p className="text-sm opacity-90">Speak for 1-2 minutes on the topic.</p>
            </div>

            {taskCard && (
              <div className="bg-white rounded-2xl shadow-sm border-2 border-blue-200 p-8">
                <h3 className="text-xl font-bold text-slate-800 mb-4">Describe {taskCard.topic}</h3>
                <p className="text-slate-700 mb-6">{taskCard.description}</p>
                
                <div className="bg-slate-50 rounded-lg p-4 border border-slate-200 mb-6">
                  <p className="text-sm font-semibold text-slate-600 mb-2">You should say:</p>
                  <ul className="list-disc list-inside space-y-1 text-sm text-slate-700">
                    {taskCard.points?.map((point: string, index: number) => (
                      <li key={index}>{point}</li>
                    ))}
                  </ul>
                </div>

                <div className="text-center">
                  {isRecording ? (
                    <div className="flex flex-col items-center gap-4">
                      <div className="w-20 h-20 bg-red-500 rounded-full flex items-center justify-center animate-pulse">
                        <div className="w-12 h-12 bg-white rounded-full"></div>
                      </div>
                      <p className="text-red-600 font-medium">Recording in progress...</p>
                      <button
                        onClick={stopRecording}
                        className="px-6 py-2 bg-slate-800 text-white rounded-lg hover:bg-slate-900"
                      >
                        Stop Recording
                      </button>
                    </div>
                  ) : (
                    <p className="text-slate-600">Recording will start automatically...</p>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {phase === 'evaluating' && (
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-12 text-center">
            <Loader2 className="animate-spin text-blue-600 mx-auto mb-4" size={32} />
            <p className="text-slate-600">Evaluating your performance...</p>
          </div>
        )}

        {phase === 'results' && evaluationResults && (
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-slate-800">Part 2 Evaluation Report</h2>
              <div className="flex items-center gap-2 bg-blue-50 px-4 py-2 rounded-lg border border-blue-100">
                <span className="text-sm text-blue-800 font-medium">Overall Band</span>
                <span className="text-2xl font-bold text-blue-600">{evaluationResults.overall_band}</span>
              </div>
            </div>

            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 mb-6">
              <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">Transcript</h3>
              <p className="text-lg leading-relaxed text-slate-700">
                "{evaluationResults.transcript}"
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              <ScoreCard 
                title="Fluency & Coherence"
                score={evaluationResults.fluency.score}
                details={evaluationResults.fluency}
                icon={Activity}
                color="bg-emerald-500"
              />
              <ScoreCard 
                title="Pronunciation"
                score={evaluationResults.pronunciation.score}
                details={evaluationResults.pronunciation}
                icon={BarChart2}
                color="bg-blue-500"
              />
              <ScoreCard 
                title="Lexical Resource"
                score={evaluationResults.vocabulary.score}
                details={evaluationResults.vocabulary}
                icon={Type}
                color="bg-purple-500"
              />
              <ScoreCard 
                title="Grammar Range"
                score={evaluationResults.grammar.score}
                details={evaluationResults.grammar}
                icon={AlertCircle}
                color="bg-amber-500"
              />
            </div>

            <button 
              onClick={handleContinue}
              className="w-full py-4 bg-blue-600 text-white font-semibold rounded-xl hover:bg-blue-700 transition-colors"
            >
              Continue to Part 3
            </button>
          </div>
        )}
      </main>
    </div>
  );
}


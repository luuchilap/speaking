import { useState, useRef, useEffect } from 'react';
import { Mic, Square, Volume2, Loader2, ArrowLeft, StopCircle, BarChart2, Activity, Type, AlertCircle, ChevronDown, ChevronUp } from 'lucide-react';

const getBackendUrl = () => {
  return import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
};

interface Part1ModeProps {
  onBack: () => void;
  onComplete: () => void;
}

export default function Part1Mode({ onBack, onComplete }: Part1ModeProps) {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState<string>('');
  const [isLoadingQuestion, setIsLoadingQuestion] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evaluationResults, setEvaluationResults] = useState<any>(null);
  const [isTestComplete, setIsTestComplete] = useState(false);
  const [userResponses, setUserResponses] = useState<Array<{ question: string; transcript: string; audioBlob: Blob }>>([]);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const userAudioBlobsRef = useRef<Blob[]>([]);

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
    loadQuestion();
  }, [currentQuestionIndex]);

  const loadQuestion = async () => {
    setIsLoadingQuestion(true);
    try {
      const backendUrl = getBackendUrl();
      const response = await fetch(`${backendUrl}/ielts/part1/question`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question_index: currentQuestionIndex }),
      });

      if (!response.ok) {
        throw new Error('Failed to load question');
      }

      const data = await response.json();
      setCurrentQuestion(data.question);
      
      // Play audio if available
      if (data.audio_url) {
        await playAudio(`${backendUrl}${data.audio_url}`);
      }
    } catch (error) {
      console.error('Error loading question:', error);
      setError(`Failed to load question: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsLoadingQuestion(false);
    }
  };

  const playAudio = async (audioUrl: string): Promise<void> => {
    return new Promise((resolve, reject) => {
      if (!audioRef.current) {
        audioRef.current = new Audio();
      }
      
      const audio = audioRef.current;
      audio.src = audioUrl;
      setIsPlayingAudio(true);

      audio.onended = () => {
        setIsPlayingAudio(false);
        resolve();
      };

      audio.onerror = () => {
        setIsPlayingAudio(false);
        reject(new Error('Failed to play audio'));
      };

      audio.play().catch((error) => {
        setIsPlayingAudio(false);
        reject(error);
      });
    });
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      chunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        userAudioBlobsRef.current.push(blob);
        await processUserAudio(blob);
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

  const processUserAudio = async (audioBlob: Blob) => {
    setIsProcessing(true);
    
    try {
      const backendUrl = getBackendUrl();
      
      const formData = new FormData();
      formData.append('file', audioBlob, 'recording.webm');

      const transcribeResponse = await fetch(`${backendUrl}/conversation/transcribe`, {
        method: 'POST',
        body: formData,
      });

      if (!transcribeResponse.ok) {
        throw new Error('Failed to transcribe audio');
      }

      const transcribeData = await transcribeResponse.json();
      const transcript = transcribeData.transcript;

      // Store response
      setUserResponses(prev => [...prev, {
        question: currentQuestion,
        transcript: transcript,
        audioBlob: audioBlob
      }]);

      // Move to next question after a short delay
      setTimeout(() => {
        if (currentQuestionIndex < 4) { // Part 1 typically has 4-5 questions
          setCurrentQuestionIndex(prev => prev + 1);
        } else {
          // Part 1 complete, show option to evaluate
          setIsTestComplete(true);
        }
      }, 1000);
    } catch (error) {
      console.error('Error processing audio:', error);
      alert('Failed to process audio. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const evaluatePart1 = async () => {
    if (userAudioBlobsRef.current.length === 0) {
      alert('No audio recordings found. Please record at least one response before evaluating.');
      return;
    }

    setIsEvaluating(true);
    
    try {
      const backendUrl = getBackendUrl();
      const formData = new FormData();
      
      userAudioBlobsRef.current.forEach((blob, index) => {
        formData.append('files', blob, `part1_audio_${index}.webm`);
      });

      const response = await fetch(`${backendUrl}/conversation/evaluate`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to evaluate');
      }

      const data = await response.json();
      setEvaluationResults(data);
    } catch (error) {
      console.error('Error evaluating:', error);
      alert(`Failed to evaluate: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsEvaluating(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900 pb-20">
      <header className="bg-white border-b border-slate-200 py-4 sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-4 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-emerald-600 rounded-lg flex items-center justify-center text-white font-bold">P1</div>
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-emerald-600 to-teal-600">
              IELTS Speaking - Part 1
            </h1>
          </div>
          <button
            onClick={onBack}
            className="text-sm font-medium text-slate-500 hover:text-emerald-600 flex items-center gap-1"
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

        {!evaluationResults ? (
          <>
            <div className="bg-emerald-600 rounded-2xl p-6 text-white shadow-lg mb-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-semibold">Part 1: Introduction & Interview</span>
                <span className="text-sm">Question {currentQuestionIndex + 1} of 5</span>
              </div>
              <p className="text-lg mt-2">{currentQuestion || 'Loading question...'}</p>
            </div>

            {isLoadingQuestion ? (
              <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-12 text-center">
                <Loader2 className="animate-spin text-emerald-600 mx-auto mb-4" size={32} />
                <p className="text-slate-600">Loading question...</p>
              </div>
            ) : (
              <>
                <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 mb-6">
                  <div className="text-center">
                    <div className="flex justify-center gap-4 mb-6">
                      {!isRecording ? (
                        <button
                          onClick={startRecording}
                          disabled={isProcessing || isPlayingAudio}
                          className="group relative flex items-center justify-center w-20 h-20 bg-red-500 rounded-full shadow-lg hover:bg-red-600 transition-all hover:scale-105 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          <Mic size={32} className="text-white" />
                          <span className="absolute -bottom-8 text-sm font-medium text-slate-500">
                            {isProcessing ? 'Processing...' : isPlayingAudio ? 'Listening...' : 'Record'}
                          </span>
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
                    {isRecording && (
                      <div className="flex items-center justify-center gap-2 text-red-500">
                        <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                        <span className="text-sm font-medium">Recording...</span>
                      </div>
                    )}
                  </div>
                </div>

                {isTestComplete && (
                  <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 text-center">
                    <p className="text-lg font-semibold text-slate-800 mb-4">Part 1 Complete!</p>
                    <p className="text-slate-600 mb-6">You've answered all questions. Click below to get your evaluation.</p>
                    <button
                      onClick={evaluatePart1}
                      disabled={isEvaluating}
                      className="px-6 py-3 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 mx-auto"
                    >
                      <StopCircle size={20} />
                      {isEvaluating ? 'Evaluating...' : 'Get Evaluation'}
                    </button>
                  </div>
                )}
              </>
            )}
          </>
        ) : (
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-slate-800">Part 1 Evaluation Report</h2>
              <div className="flex items-center gap-2 bg-emerald-50 px-4 py-2 rounded-lg border border-emerald-100">
                <span className="text-sm text-emerald-800 font-medium">Overall Band</span>
                <span className="text-2xl font-bold text-emerald-600">{evaluationResults.overall_band}</span>
              </div>
            </div>

            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 mb-6">
              <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">Full Transcript</h3>
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
              onClick={onComplete}
              className="w-full py-4 bg-white border border-slate-200 text-slate-600 font-semibold rounded-xl hover:bg-slate-50 transition-colors"
            >
              Return to Test Selection
            </button>
          </div>
        )}
      </main>
    </div>
  );
}


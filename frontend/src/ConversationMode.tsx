import { useState, useRef, useEffect } from 'react';
import { Mic, Square, RefreshCw, Volume2, Loader2, MessageSquare } from 'lucide-react';

const getBackendUrl = () => {
  return import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
};

interface ConversationMessage {
  role: 'assistant' | 'user';
  content: string;
  audioUrl?: string;
  timestamp: Date;
}

interface ConversationModeProps {
  onBack: () => void;
}

export default function ConversationMode({ onBack }: ConversationModeProps) {
  console.log('ConversationMode component rendering');
  
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);
  const [availableTopics, setAvailableTopics] = useState<string[]>([]);
  const [isLoadingTopics, setIsLoadingTopics] = useState(false);
  const [showTopicSelector, setShowTopicSelector] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const loadTopics = async () => {
    setIsLoadingTopics(true);
    try {
      const backendUrl = getBackendUrl();
      const response = await fetch(`${backendUrl}/conversation/topics`);
      if (response.ok) {
        const data = await response.json();
        setAvailableTopics(data.topics || []);
      } else {
        console.error('Failed to load topics:', response.status, response.statusText);
        // Fallback topics if API fails
        setAvailableTopics(['Music', 'Sports', 'Travel', 'Food', 'Hobbies']);
      }
    } catch (error) {
      console.error('Failed to load topics:', error);
      setError(`Failed to load topics: ${error instanceof Error ? error.message : 'Unknown error'}`);
      // Fallback topics if API fails
      setAvailableTopics(['Music', 'Sports', 'Travel', 'Food', 'Hobbies']);
    } finally {
      setIsLoadingTopics(false);
    }
  };

  // Load available topics on mount
  useEffect(() => {
    loadTopics();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const startConversation = async (topic: string) => {
    setIsProcessing(true);
    try {
      const backendUrl = getBackendUrl();
      const response = await fetch(`${backendUrl}/conversation/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ topic }),
      });

      if (!response.ok) {
        throw new Error('Failed to start conversation');
      }

      const data = await response.json();
      setConversationId(data.conversation_id);
      setSelectedTopic(topic);
      setShowTopicSelector(false);

      // Add assistant message
      const assistantMessage: ConversationMessage = {
        role: 'assistant',
        content: data.message,
        audioUrl: data.audio_url ? `${backendUrl}${data.audio_url}` : undefined,
        timestamp: new Date(),
      };
      setMessages([assistantMessage]);

      // Play audio if available
      if (data.audio_url) {
        await playAudio(`${backendUrl}${data.audio_url}`);
      }
    } catch (error) {
      console.error('Error starting conversation:', error);
      alert('Failed to start conversation. Please try again.');
    } finally {
      setIsProcessing(false);
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
      
      // First, transcribe the audio
      const formData = new FormData();
      formData.append('file', audioBlob, 'recording.webm');
      if (conversationId) {
        formData.append('conversation_id', conversationId);
      }

      const transcribeResponse = await fetch(`${backendUrl}/conversation/transcribe`, {
        method: 'POST',
        body: formData,
      });

      if (!transcribeResponse.ok) {
        throw new Error('Failed to transcribe audio');
      }

      const transcribeData = await transcribeResponse.json();
      const transcript = transcribeData.transcript;

      // Add user message to conversation
      const userMessage: ConversationMessage = {
        role: 'user',
        content: transcript,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMessage]);

      // Get AI response
      const respondResponse = await fetch(`${backendUrl}/conversation/respond`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          conversation_id: conversationId,
          user_response: transcript,
        }),
      });

      if (!respondResponse.ok) {
        throw new Error('Failed to get AI response');
      }

      const respondData = await respondResponse.json();
      
      // Update conversation ID if it was returned
      if (respondData.conversation_id && !conversationId) {
        setConversationId(respondData.conversation_id);
      }

      // Add assistant response
      const assistantMessage: ConversationMessage = {
        role: 'assistant',
        content: respondData.message,
        audioUrl: respondData.audio_url ? `${backendUrl}${respondData.audio_url}` : undefined,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMessage]);

      // Play AI audio response
      if (respondData.audio_url) {
        await playAudio(`${backendUrl}${respondData.audio_url}`);
      }
    } catch (error) {
      console.error('Error processing audio:', error);
      alert('Failed to process audio. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const resetConversation = () => {
    setConversationId(null);
    setMessages([]);
    setSelectedTopic(null);
    setShowTopicSelector(true);
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    setIsPlayingAudio(false);
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900 pb-20">
      <header className="bg-white border-b border-slate-200 py-4 sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-4 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center text-white font-bold">AI</div>
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-purple-600">
              IELTS Conversation Practice
            </h1>
          </div>
          <div className="flex gap-2">
            <button
              onClick={onBack}
              className="text-sm font-medium text-slate-500 hover:text-indigo-600"
            >
              Single Recording
            </button>
            {messages.length > 0 && (
              <button
                onClick={resetConversation}
                className="text-sm font-medium text-slate-500 hover:text-indigo-600 flex items-center gap-1"
              >
                <RefreshCw size={16} />
                New Conversation
              </button>
            )}
          </div>
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
        {showTopicSelector ? (
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-8">
            <h2 className="text-2xl font-bold text-slate-800 mb-4">Choose a Conversation Topic</h2>
            <p className="text-slate-600 mb-6">Select a topic to start practicing your IELTS speaking conversation.</p>
            
            {isLoadingTopics ? (
              <div className="flex justify-center py-8">
                <Loader2 className="animate-spin text-indigo-600" size={32} />
              </div>
            ) : availableTopics.length === 0 ? (
              <div className="text-center py-8">
                <p className="text-slate-500 mb-4">No topics available. Please check your backend connection.</p>
                <button
                  onClick={loadTopics}
                  className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
                >
                  Retry
                </button>
              </div>
            ) : (
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {availableTopics.map((topic) => (
                  <button
                    key={topic}
                    onClick={() => startConversation(topic)}
                    disabled={isProcessing}
                    className="p-4 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200 rounded-lg text-indigo-800 font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {topic}
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          <>
            {selectedTopic && (
              <div className="bg-indigo-600 rounded-2xl p-4 text-white shadow-lg mb-6">
                <span className="text-sm font-semibold">Topic: {selectedTopic}</span>
              </div>
            )}

            {/* Conversation Messages */}
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 mb-6 max-h-[500px] overflow-y-auto">
              {messages.length === 0 ? (
                <div className="text-center text-slate-500 py-8">
                  <MessageSquare size={48} className="mx-auto mb-4 text-slate-300" />
                  <p>Conversation will appear here...</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {messages.map((message, index) => (
                    <div
                      key={index}
                      className={`flex ${message.role === 'assistant' ? 'justify-start' : 'justify-end'}`}
                    >
                      <div
                        className={`max-w-[75%] rounded-2xl p-4 ${
                          message.role === 'assistant'
                            ? 'bg-indigo-50 text-slate-800 border border-indigo-100'
                            : 'bg-indigo-600 text-white'
                        }`}
                      >
                        <div className="flex items-start gap-2">
                          {message.role === 'assistant' && message.audioUrl && (
                            <button
                              onClick={() => playAudio(message.audioUrl!)}
                              disabled={isPlayingAudio}
                              className="mt-1 p-1 hover:bg-indigo-100 rounded transition-colors disabled:opacity-50"
                            >
                              <Volume2 size={16} className="text-indigo-600" />
                            </button>
                          )}
                          <p className="text-sm leading-relaxed">{message.content}</p>
                        </div>
                        <span className="text-xs opacity-70 mt-2 block">
                          {message.timestamp.toLocaleTimeString()}
                        </span>
                      </div>
                    </div>
                  ))}
                  {isProcessing && (
                    <div className="flex justify-start">
                      <div className="bg-slate-100 rounded-2xl p-4">
                        <Loader2 className="animate-spin text-indigo-600" size={20} />
                      </div>
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>

            {/* Recording Controls */}
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 text-center">
              <div className="flex justify-center gap-4">
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
                <div className="mt-4 flex items-center justify-center gap-2 text-red-500">
                  <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                  <span className="text-sm font-medium">Recording...</span>
                </div>
              )}
            </div>
          </>
        )}
      </main>
    </div>
  );
}


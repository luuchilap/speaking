import { useState } from 'react';
import { BookOpen, Clock, MessageSquare, ArrowLeft } from 'lucide-react';
import Part1Mode from './Part1Mode';
import Part2Mode from './Part2Mode';
import Part3Mode from './Part3Mode';

interface IELTSTestModeProps {
  onBack: () => void;
}

type TestPart = 'select' | 'part1' | 'part2' | 'part3';

export default function IELTSTestMode({ onBack }: IELTSTestModeProps) {
  const [selectedPart, setSelectedPart] = useState<TestPart>('select');
  const [part2Topic, setPart2Topic] = useState<string | null>(null);

  if (selectedPart === 'part1') {
    return <Part1Mode onBack={() => setSelectedPart('select')} onComplete={() => setSelectedPart('select')} />;
  }

  if (selectedPart === 'part2') {
    return <Part2Mode 
      onBack={() => setSelectedPart('select')} 
      onComplete={(topic) => {
        setPart2Topic(topic);
        setSelectedPart('part3');
      }} 
    />;
  }

  if (selectedPart === 'part3') {
    return <Part3Mode 
      onBack={() => setSelectedPart('part2')} 
      onComplete={() => setSelectedPart('select')}
      part2Topic={part2Topic || ''}
    />;
  }

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900">
      <header className="bg-white border-b border-slate-200 py-4 sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-4 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center text-white font-bold">AI</div>
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-purple-600">
              IELTS Speaking Test
            </h1>
          </div>
          <button
            onClick={onBack}
            className="text-sm font-medium text-slate-500 hover:text-indigo-600 flex items-center gap-1"
          >
            <ArrowLeft size={16} />
            Back
          </button>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 mt-12">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-slate-800 mb-4">Choose a Test Part</h2>
          <p className="text-slate-600 text-lg">
            Select which part of the IELTS Speaking test you'd like to practice
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6 mb-8">
          {/* Part 1 */}
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-8 hover:shadow-md transition-shadow cursor-pointer"
               onClick={() => setSelectedPart('part1')}>
            <div className="flex items-center justify-center w-16 h-16 bg-emerald-100 rounded-full mb-4 mx-auto">
              <MessageSquare className="text-emerald-600" size={32} />
            </div>
            <h3 className="text-xl font-bold text-slate-800 mb-2 text-center">Part 1</h3>
            <p className="text-sm text-slate-500 mb-4 text-center">Introduction & Interview</p>
            <div className="space-y-2 text-sm text-slate-600">
              <div className="flex items-center gap-2">
                <Clock size={16} className="text-slate-400" />
                <span>4-5 minutes</span>
              </div>
              <p className="text-xs text-slate-500">
                Answer general questions about familiar topics like home, family, work, and interests.
              </p>
            </div>
            <button className="mt-6 w-full py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition-colors font-medium">
            Start Part 1
            </button>
          </div>

          {/* Part 2 */}
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-8 hover:shadow-md transition-shadow cursor-pointer"
               onClick={() => setSelectedPart('part2')}>
            <div className="flex items-center justify-center w-16 h-16 bg-blue-100 rounded-full mb-4 mx-auto">
              <BookOpen className="text-blue-600" size={32} />
            </div>
            <h3 className="text-xl font-bold text-slate-800 mb-2 text-center">Part 2</h3>
            <p className="text-sm text-slate-500 mb-4 text-center">Long Turn</p>
            <div className="space-y-2 text-sm text-slate-600">
              <div className="flex items-center gap-2">
                <Clock size={16} className="text-slate-400" />
                <span>3-4 minutes</span>
              </div>
              <p className="text-xs text-slate-500">
                Prepare for 1 minute, then speak for 1-2 minutes on a given topic.
              </p>
            </div>
            <button className="mt-6 w-full py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium">
              Start Part 2
            </button>
          </div>

          {/* Part 3 */}
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-8 hover:shadow-md transition-shadow cursor-pointer"
               onClick={() => {
                 if (!part2Topic) {
                   alert('Please complete Part 2 first to get a topic for Part 3 discussion.');
                   return;
                 }
                 setSelectedPart('part3');
               }}>
            <div className="flex items-center justify-center w-16 h-16 bg-purple-100 rounded-full mb-4 mx-auto">
              <MessageSquare className="text-purple-600" size={32} />
            </div>
            <h3 className="text-xl font-bold text-slate-800 mb-2 text-center">Part 3</h3>
            <p className="text-sm text-slate-500 mb-4 text-center">Discussion</p>
            <div className="space-y-2 text-sm text-slate-600">
              <div className="flex items-center gap-2">
                <Clock size={16} className="text-slate-400" />
                <span>4-5 minutes</span>
              </div>
              <p className="text-xs text-slate-500">
                Discuss abstract ideas and issues related to the Part 2 topic.
              </p>
            </div>
            {part2Topic ? (
              <button className="mt-6 w-full py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors font-medium">
                Continue to Part 3
              </button>
            ) : (
              <button className="mt-6 w-full py-2 bg-slate-300 text-slate-500 rounded-lg cursor-not-allowed font-medium" disabled>
                Complete Part 2 First
              </button>
            )}
          </div>
        </div>

        <div className="bg-indigo-50 border border-indigo-200 rounded-xl p-6">
          <h4 className="font-semibold text-indigo-800 mb-2">Test Structure</h4>
          <p className="text-sm text-indigo-700">
            The IELTS Speaking test consists of three parts. You can practice each part individually, 
            or complete them in sequence (Part 2 → Part 3) for a more realistic test experience.
          </p>
        </div>
      </main>
    </div>
  );
}


import { useState } from 'react';
import { X, Save, FileText } from 'lucide-react';

interface SaveFeedbackModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (note: string) => void;
  feedbackTitle: string;
  feedbackData: {
    title: string;
    score: number;
    details: any;
    overallBand?: number;
    transcript?: string;
  };
}

export default function SaveFeedbackModal({
  isOpen,
  onClose,
  onSave,
  feedbackTitle,
  feedbackData,
}: SaveFeedbackModalProps) {
  const [note, setNote] = useState('');
  const [isSaving, setIsSaving] = useState(false);

  if (!isOpen) return null;

  const handleSave = async () => {
    if (isSaving) return;
    setIsSaving(true);
    // Small delay for better UX
    setTimeout(() => {
      onSave(note);
      setNote('');
      setIsSaving(false);
      onClose();
    }, 300);
  };

  const handleClose = () => {
    if (!isSaving) {
      setNote('');
      onClose();
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4">
      <div className="bg-white rounded-2xl shadow-xl max-w-lg w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-200">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-indigo-100 rounded-lg">
              <Save size={20} className="text-indigo-600" />
            </div>
            <h2 className="text-xl font-bold text-slate-800">Save for Future Review</h2>
          </div>
          <button
            onClick={handleClose}
            disabled={isSaving}
            className="p-2 hover:bg-slate-100 rounded-lg transition-colors disabled:opacity-50"
          >
            <X size={20} className="text-slate-500" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto flex-1">
          {/* Feedback Summary */}
          <div className="bg-slate-50 rounded-xl p-4 mb-4 border border-slate-200">
            <div className="flex items-center gap-2 mb-2">
              <FileText size={16} className="text-slate-500" />
              <span className="text-sm font-semibold text-slate-600">Feedback Summary</span>
            </div>
            <div className="space-y-1 text-sm text-slate-700">
              <p><span className="font-medium">Category:</span> {feedbackData.title}</p>
              <p><span className="font-medium">Score:</span> {feedbackData.score.toFixed(1)}</p>
              {feedbackData.overallBand && (
                <p><span className="font-medium">Overall Band:</span> {feedbackData.overallBand.toFixed(1)}</p>
              )}
              {feedbackData.transcript && (
                <p className="mt-2 text-xs text-slate-600 line-clamp-2">
                  <span className="font-medium">Transcript:</span> {feedbackData.transcript.substring(0, 100)}...
                </p>
              )}
            </div>
          </div>

          {/* Note Input */}
          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-2">
              Add a Note (Optional)
            </label>
            <textarea
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="Add your thoughts, what you learned, or areas to focus on..."
              className="w-full h-32 p-3 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-none text-sm text-slate-700 placeholder-slate-400"
              disabled={isSaving}
            />
            <p className="mt-1 text-xs text-slate-500">
              This note will help you remember what to focus on during future practice.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 p-6 border-t border-slate-200 bg-slate-50">
          <button
            onClick={handleClose}
            disabled={isSaving}
            className="px-4 py-2 text-slate-600 font-medium hover:bg-slate-200 rounded-lg transition-colors disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={isSaving}
            className="px-6 py-2 bg-indigo-600 text-white font-medium rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {isSaving ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                Saving...
              </>
            ) : (
              <>
                <Save size={16} />
                Save
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}


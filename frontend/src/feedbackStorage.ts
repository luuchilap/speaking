export type SavedFeedback = {
  id: string;
  timestamp: number;
  date: string;
  note: string;
  feedbackData: {
    title: string;
    score: number;
    details: any;
    overallBand?: number;
    transcript?: string;
  };
};

const STORAGE_KEY = 'ielts_saved_feedback';

export const saveFeedback = (feedback: Omit<SavedFeedback, 'id' | 'timestamp' | 'date'>): SavedFeedback => {
  const savedFeedback: SavedFeedback = {
    ...feedback,
    id: `feedback_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    timestamp: Date.now(),
    date: new Date().toISOString(),
  };

  const existing = getSavedFeedback();
  existing.unshift(savedFeedback); // Add to beginning (most recent first)
  
  // Limit to last 50 saved feedbacks
  const limited = existing.slice(0, 50);
  
  localStorage.setItem(STORAGE_KEY, JSON.stringify(limited));
  return savedFeedback;
};

export const getSavedFeedback = (): SavedFeedback[] => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return [];
    return JSON.parse(stored);
  } catch (error) {
    console.error('Error loading saved feedback:', error);
    return [];
  }
};

export const deleteSavedFeedback = (id: string): void => {
  const existing = getSavedFeedback();
  const filtered = existing.filter((item) => item.id !== id);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
};

export const clearAllSavedFeedback = (): void => {
  localStorage.removeItem(STORAGE_KEY);
};


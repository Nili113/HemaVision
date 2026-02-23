import { useState, useCallback } from 'react';
import { predictMultiCell, fileToBase64, type MultiCellResponse } from '../lib/api';

interface UseAnalysisReturn {
  result: MultiCellResponse | null;
  isLoading: boolean;
  error: string | null;
  analyze: (file: File) => Promise<void>;
  reset: () => void;
}

export function useAnalysis(): UseAnalysisReturn {
  const [result, setResult] = useState<MultiCellResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const analyze = useCallback(async (file: File) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const base64 = await fileToBase64(file);
      const response = await predictMultiCell({ image_base64: base64 });
      setResult(response);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Analysis failed. Please try again.';
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
    setIsLoading(false);
  }, []);

  return { result, isLoading, error, analyze, reset };
}

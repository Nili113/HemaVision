import { useState, useCallback } from 'react';
import axios from 'axios';
import { predictMultiCell, fileToBase64, type MultiCellResponse } from '../lib/api';

export type SegmentationMode = 'auto' | 'single' | 'multi';

interface UseAnalysisReturn {
  result: MultiCellResponse | null;
  isLoading: boolean;
  error: string | null;
  analyze: (file: File, segmentationMode?: SegmentationMode) => Promise<void>;
  reset: () => void;
}

export function useAnalysis(): UseAnalysisReturn {
  const [result, setResult] = useState<MultiCellResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const analyze = useCallback(async (file: File, segmentationMode: SegmentationMode = 'auto') => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const base64 = await fileToBase64(file);
      const response = await predictMultiCell({ image_base64: base64, segmentation_mode: segmentationMode });
      setResult(response);
    } catch (err) {
      let message = 'Analysis failed. Please try again.';

      if (axios.isAxiosError(err)) {
        if (!err.response) {
          message = 'Backend is not running. Start FastAPI on port 8000 and try again.';
        } else {
          message = err.response.data?.detail || err.message || message;
        }
      } else if (err instanceof Error) {
        message = err.message;
      }

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

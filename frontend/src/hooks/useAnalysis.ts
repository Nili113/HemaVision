import { useState, useCallback } from 'react';
import { predict, fileToBase64, type PredictionResponse } from '../lib/api';

interface PatientData {
  age: number;
  sex: string;
  npm1_mutated: boolean;
  flt3_mutated: boolean;
  genetic_other: boolean;
}

interface UseAnalysisReturn {
  result: PredictionResponse | null;
  isLoading: boolean;
  error: string | null;
  analyze: (file: File, patientData: PatientData) => Promise<void>;
  reset: () => void;
}

export function useAnalysis(): UseAnalysisReturn {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const analyze = useCallback(async (file: File, patientData: PatientData) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const base64 = await fileToBase64(file);
      const response = await predict({
        image_base64: base64,
        ...patientData,
      });
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

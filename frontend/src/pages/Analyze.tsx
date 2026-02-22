import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { RotateCcw, Loader2 } from 'lucide-react';
import clsx from 'clsx';
import UploadZone from '../components/UploadZone';
import ResultCard from '../components/ResultCard';
import GradCAMViewer from '../components/GradCAMViewer';
import { useAnalysis } from '../hooks/useAnalysis';

const stepLabels = ['Upload Image', 'Processing', 'Results'];

export default function Analyze() {
  const [step, setStep] = useState(1);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const { result, isLoading, error, analyze, reset } = useAnalysis();

  const handleImageUpload = useCallback(
    async (file: File) => {
      setImageFile(file);
      const reader = new FileReader();
      reader.onload = (e) => setImagePreview(e.target?.result as string);
      reader.readAsDataURL(file);
      // Go straight to results — no patient form needed
      setStep(2);
      await analyze(file);
      setStep(3);
    },
    [analyze]
  );

  const handleReset = useCallback(() => {
    setStep(1);
    setImageFile(null);
    setImagePreview(null);
    reset();
  }, [reset]);

  return (
    <div className="min-h-screen py-8">
      <div className="section-container space-y-8">
        {/* ── Header ─────────────────────────────────── */}
        <header>
          <div className="flex flex-col md:flex-row md:items-start justify-between gap-6 mb-8">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <div className="w-10 h-10 rounded-xl flex items-center justify-center"
                  style={{ background: 'linear-gradient(135deg, rgba(19,127,236,0.15) 0%, rgba(124,58,237,0.1) 100%)' }}>
                  <span className="material-icons-outlined text-primary text-xl">biotech</span>
                </div>
                <div>
                  <h1 className="text-xl font-bold text-white leading-tight">New Diagnostic Session</h1>
                  <p className="text-slate-500 text-xs mt-0.5">Upload a blood slide for AML screening</p>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-4">
              {/* Stepper — pill style */}
              <div className="flex items-center gap-1 p-1 rounded-xl" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
                {stepLabels.map((label, idx) => {
                  const stepNum = idx + 1;
                  const isDone = step > stepNum;
                  const isCurrent = step === stepNum;
                  return (
                    <div key={label} className="flex items-center">
                      <div
                        className={clsx(
                          'flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold transition-all',
                          isCurrent
                            ? 'bg-primary/15 text-primary'
                            : isDone
                            ? 'text-emerald-400'
                            : 'text-slate-600'
                        )}
                      >
                        <div
                          className={clsx(
                            'w-5 h-5 rounded-full text-[10px] font-bold flex items-center justify-center',
                            isCurrent
                              ? 'bg-primary text-white'
                              : isDone
                              ? 'bg-emerald-500/20 text-emerald-400'
                              : 'border border-slate-700 text-slate-600'
                          )}
                        >
                          {isDone ? '✓' : stepNum}
                        </div>
                        <span className="hidden sm:inline">{label}</span>
                      </div>
                      {idx < stepLabels.length - 1 && (
                        <div
                          className={clsx(
                            'w-5 h-[1px] mx-0.5',
                            step > stepNum ? 'bg-emerald-500/40' : 'bg-slate-800'
                          )}
                        />
                      )}
                    </div>
                  );
                })}
              </div>

              {step > 1 && (
                <button
                  onClick={handleReset}
                  className="flex items-center gap-1.5 text-xs font-medium text-slate-400 hover:text-white px-3 py-2 rounded-lg border border-slate-800 hover:border-slate-700 transition-all"
                >
                  <RotateCcw size={12} />
                  Reset
                </button>
              )}
            </div>
          </div>
        </header>

        {/* ── Content ────────────────────────────────── */}
        <AnimatePresence mode="wait">
          {/* Step 1: Upload image */}
          {step === 1 && (
            <motion.div
              key="upload"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.3 }}
            >
              <UploadZone onUpload={handleImageUpload} />
            </motion.div>
          )}

          {/* Step 2: Processing / Step 3: Results */}
          {step >= 2 && (
            <motion.div
              key="results"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.3 }}
              className="space-y-6"
            >
              {/* Image thumbnail */}
              {imagePreview && (
                <div className="flex items-center gap-4 p-4 rounded-xl bg-surface border border-slate-800">
                  <img
                    src={imagePreview}
                    alt="Uploaded cell"
                    className="w-14 h-14 rounded-lg object-cover border border-slate-700"
                  />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-white truncate">
                      {imageFile?.name}
                    </div>
                    <div className="text-xs text-slate-500">
                      {imageFile && (imageFile.size / 1024).toFixed(1)} KB
                    </div>
                  </div>
                </div>
              )}

              {/* Loading */}
              {isLoading && (
                <div className="text-center py-24">
                  <div className="inline-flex items-center justify-center w-14 h-14 rounded-full bg-primary/10 mb-5">
                    <Loader2 className="w-6 h-6 text-primary animate-spin" />
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-1.5">Analyzing blood slide...</h3>
                  <p className="text-sm text-slate-400">
                    Running dual-stream fusion with Grad-CAM
                  </p>
                </div>
              )}

              {/* Error */}
              {error && (
                <div className="text-center py-16">
                  <p className="text-base text-red-400 font-medium mb-2">Analysis Failed</p>
                  <p className="text-sm text-slate-400 mb-6">{error}</p>
                  <button onClick={handleReset} className="btn-primary">
                    Try Again
                  </button>
                </div>
              )}

              {/* Results */}
              {result && (
                <>
                  <ResultCard result={result} />
                  <GradCAMViewer
                    originalImage={imagePreview}
                    gradcamBase64={result.gradcam_base64}
                  />
                  <div className="flex justify-center pt-4">
                    <button
                      onClick={handleReset}
                      className="btn-ghost inline-flex items-center gap-2"
                    >
                      <RotateCcw size={14} />
                      Analyze another cell
                    </button>
                  </div>
                </>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

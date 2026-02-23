import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { RotateCcw, Loader2 } from 'lucide-react';
import clsx from 'clsx';
import UploadZone from '../components/UploadZone';
import ResultCard from '../components/ResultCard';
import GradCAMViewer from '../components/GradCAMViewer';
import { useAnalysis } from '../hooks/useAnalysis';

const stepLabels = ['Upload Image', 'Processing', 'Results'];

/**
 * Convert any image file to a displayable data URL.
 * Handles TIFF and other formats browsers can't natively display
 * by drawing them to a canvas via an offscreen Image element.
 */
function fileToPreviewUrl(file: File): Promise<string> {
  return new Promise((resolve) => {
    // For common formats, FileReader is fine
    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target?.result as string;
      // Try loading as <img> to test browser support
      const img = new window.Image();
      img.onload = () => {
        // If it loads, the browser can display it
        resolve(dataUrl);
      };
      img.onerror = () => {
        // Browser can't display this format (e.g. TIFF) — use a placeholder
        resolve('');
      };
      img.src = dataUrl;
    };
    reader.onerror = () => resolve('');
    reader.readAsDataURL(file);
  });
}

export default function Analyze() {
  const [step, setStep] = useState(1);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const { result, isLoading, error, analyze, reset } = useAnalysis();

  const handleImageUpload = useCallback(
    async (file: File) => {
      setImageFile(file);
      // Generate preview — handles TIFF gracefully
      const preview = await fileToPreviewUrl(file);
      setImagePreview(preview || null);
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

  // Derive a file icon based on extension
  const fileExtIcon = imageFile?.name?.match(/\.(tiff?|bmp)$/i) ? 'description' : 'image';

  return (
    <div className="min-h-screen py-8">
      <div className="section-container space-y-6">
        {/* ── Header ─────────────────────────────────── */}
        <header>
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl flex items-center justify-center"
                style={{ background: 'linear-gradient(135deg, rgba(19,127,236,0.15) 0%, rgba(124,58,237,0.1) 100%)' }}>
                <span className="material-icons-outlined text-primary text-xl">biotech</span>
              </div>
              <div>
                <h1 className="text-xl font-bold text-white leading-tight">New Diagnostic Session</h1>
                <p className="text-slate-500 text-xs mt-0.5">Upload a blood slide for AML screening</p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              {/* Stepper */}
              <div className="flex items-center gap-0.5 p-1 rounded-xl" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
                {stepLabels.map((label, idx) => {
                  const stepNum = idx + 1;
                  const isDone = step > stepNum;
                  const isCurrent = step === stepNum;
                  return (
                    <div key={label} className="flex items-center">
                      <div
                        className={clsx(
                          'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all',
                          isCurrent ? 'bg-primary/15 text-primary' : isDone ? 'text-emerald-400' : 'text-slate-600'
                        )}
                      >
                        <div
                          className={clsx(
                            'w-5 h-5 rounded-full text-[10px] font-bold flex items-center justify-center',
                            isCurrent ? 'bg-primary text-white' : isDone ? 'bg-emerald-500/20 text-emerald-400' : 'border border-slate-700 text-slate-600'
                          )}
                        >
                          {isDone ? '✓' : stepNum}
                        </div>
                        <span className="hidden sm:inline">{label}</span>
                      </div>
                      {idx < stepLabels.length - 1 && (
                        <div className={clsx('w-4 h-px mx-0.5', step > stepNum ? 'bg-emerald-500/40' : 'bg-slate-800')} />
                      )}
                    </div>
                  );
                })}
              </div>

              {step > 1 && (
                <button
                  onClick={handleReset}
                  className="flex items-center gap-1.5 text-xs font-medium text-slate-400 hover:text-white px-3 py-2 rounded-lg border border-slate-800 hover:border-slate-600 transition-all"
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
          {step === 1 && (
            <motion.div key="upload" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.25 }}>
              <UploadZone onUpload={handleImageUpload} />
            </motion.div>
          )}

          {step >= 2 && (
            <motion.div key="results" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.25 }} className="space-y-5">

              {/* ── File info bar ────────────────────── */}
              {imageFile && (
                <div className="flex items-center gap-3 px-4 py-3 rounded-xl border border-slate-800" style={{ background: '#141d27' }}>
                  {imagePreview ? (
                    <img src={imagePreview} alt="" className="w-11 h-11 rounded-lg object-cover border border-slate-700" />
                  ) : (
                    <div className="w-11 h-11 rounded-lg flex items-center justify-center border border-slate-700" style={{ background: 'rgba(19,127,236,0.08)' }}>
                      <span className="material-icons-outlined text-primary text-lg">{fileExtIcon}</span>
                    </div>
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-white truncate">{imageFile.name}</p>
                    <p className="text-xs text-slate-500">{(imageFile.size / 1024).toFixed(1)} KB</p>
                  </div>
                  {result && (
                    <div className={clsx(
                      'text-xs font-semibold px-2.5 py-1 rounded-md',
                      result.blast_count === 0 ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' :
                      result.overall_risk_level === 'HIGH RISK' ? 'bg-red-500/10 text-red-400 border border-red-500/20' :
                      'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                    )}>
                      {result.blast_count === 0 ? 'Normal' : result.overall_risk_level === 'HIGH RISK' ? 'AML Detected' : 'Atypical'}
                    </div>
                  )}
                </div>
              )}

              {/* ── Loading state ────────────────────── */}
              {isLoading && (
                <div className="text-center py-20">
                  <div className="relative inline-flex items-center justify-center w-16 h-16 mb-5">
                    <div className="absolute inset-0 rounded-full bg-primary/10 animate-ping" style={{ animationDuration: '2s' }} />
                    <div className="relative w-14 h-14 rounded-full bg-primary/10 flex items-center justify-center">
                      <Loader2 className="w-6 h-6 text-primary animate-spin" />
                    </div>
                  </div>
                  <h3 className="text-base font-semibold text-white mb-1">Analyzing blood slide...</h3>
                  <p className="text-sm text-slate-500">
                    Segmentation → Morphology → ResNet-50 → Grad-CAM
                  </p>
                  <div className="mt-4 flex justify-center gap-1">
                    {[0, 1, 2, 3].map(i => (
                      <div key={i} className="w-1.5 h-1.5 rounded-full bg-primary/40 animate-pulse" style={{ animationDelay: `${i * 200}ms` }} />
                    ))}
                  </div>
                </div>
              )}

              {/* ── Error state ──────────────────────── */}
              {error && (
                <div className="text-center py-16">
                  <div className="inline-flex items-center justify-center w-14 h-14 rounded-full bg-red-500/10 mb-4">
                    <span className="material-icons-outlined text-red-400 text-2xl">error_outline</span>
                  </div>
                  <p className="text-base text-red-400 font-semibold mb-1.5">Analysis Failed</p>
                  <p className="text-sm text-slate-500 mb-6 max-w-sm mx-auto">{error}</p>
                  <button onClick={handleReset} className="btn-primary">
                    Try Again
                  </button>
                </div>
              )}

              {/* ── Results ──────────────────────────── */}
              {result && (
                <div className="space-y-5">
                  <ResultCard result={result} />

                  {/* Annotated segmentation map (multi-cell) */}
                  {result.annotated_image_base64 && result.is_multi_cell && (
                    <div className="rounded-xl border border-slate-800 bg-surface p-6">
                      <div className="flex items-center gap-3 mb-5">
                        <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
                          <span className="material-icons-outlined text-primary text-base">grid_view</span>
                        </div>
                        <div>
                          <h3 className="text-sm font-semibold text-white">Cell Segmentation Map</h3>
                          <p className="text-xs text-slate-500">{result.num_cells} cells detected</p>
                        </div>
                      </div>
                      <div className="bg-slate-900 rounded-lg overflow-hidden border border-slate-800">
                        <img src={`data:image/png;base64,${result.annotated_image_base64}`} alt="Segmented cells" className="w-full h-auto object-contain" />
                      </div>
                    </div>
                  )}

                  {/* Per-cell Grad-CAM gallery (multi-cell) */}
                  {result.is_multi_cell && result.cells.some(c => c.gradcam_base64) && (
                    <div className="rounded-xl border border-slate-800 bg-surface p-6">
                      <div className="flex items-center gap-3 mb-5">
                        <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
                          <span className="material-icons-outlined text-primary text-base">visibility</span>
                        </div>
                        <div>
                          <h3 className="text-sm font-semibold text-white">Per-Cell Grad-CAM</h3>
                          <p className="text-xs text-slate-500">Activation heatmaps for each segmented cell</p>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                        {result.cells.map((cell) => {
                          if (!cell.gradcam_base64) return null;
                          const cellIsBlast = cell.prediction.includes('Blast');
                          return (
                            <div key={cell.cell_index} className="relative group">
                              <div className={clsx('rounded-lg overflow-hidden border-2 transition-all', cellIsBlast ? 'border-red-500/30 hover:border-red-500/60' : 'border-emerald-500/30 hover:border-emerald-500/60')}>
                                <img src={`data:image/png;base64,${cell.gradcam_base64}`} alt={`Cell ${cell.cell_index}`} className="w-full aspect-square object-contain bg-slate-900" />
                              </div>
                              <div className="absolute top-1.5 left-1.5">
                                <span className={clsx('text-[10px] font-bold px-1.5 py-0.5 rounded', cellIsBlast ? 'bg-red-500 text-white' : 'bg-emerald-500 text-white')}>
                                  #{cell.cell_index}
                                </span>
                              </div>
                              <p className={clsx('text-center mt-1 text-xs font-medium', cellIsBlast ? 'text-red-400' : 'text-emerald-400')}>
                                {cellIsBlast ? 'Blast' : 'Normal'} — {(cell.confidence * 100).toFixed(0)}%
                              </p>
                            </div>
                          );
                        })}
                      </div>
                      <div className="mt-4 flex items-center justify-between">
                        <span className="text-[10px] font-medium text-slate-600 uppercase tracking-wider">Activation Intensity</span>
                        <div className="flex items-center gap-1.5">
                          <span className="text-[10px] text-slate-600">Low</span>
                          <div className="w-20 h-1 rounded-full bg-gradient-to-r from-blue-500 via-green-500 via-yellow-400 to-red-500" />
                          <span className="text-[10px] text-slate-600">High</span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Single-cell: standard GradCAM viewer */}
                  {!result.is_multi_cell && (
                    <GradCAMViewer originalImage={imagePreview} gradcamBase64={result.cells[0]?.gradcam_base64 ?? null} />
                  )}

                  {/* Reset button */}
                  <div className="flex justify-center pt-2 pb-4">
                    <button onClick={handleReset} className="btn-ghost inline-flex items-center gap-2 text-sm">
                      <RotateCcw size={14} />
                      Analyze another slide
                    </button>
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

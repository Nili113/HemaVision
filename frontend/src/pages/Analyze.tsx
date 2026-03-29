import { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { RotateCcw, Loader2, ShieldAlert } from 'lucide-react';
import clsx from 'clsx';
import UploadZone from '../components/UploadZone';
import ResultCard from '../components/ResultCard';
import GradCAMViewer from '../components/GradCAMViewer';
import { useAnalysis, type SegmentationMode } from '../hooks/useAnalysis';
import { useAuth } from '../contexts/AuthContext';
import { getGuestUsage, incrementGuestUsage, type GuestUsage } from '../lib/guestUsage';

const stepLabels = ['Upload Image', 'Processing', 'Results'];

function fileToPreviewUrl(file: File): Promise<string> {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target?.result as string;
      const img = new window.Image();
      img.onload = () => resolve(dataUrl);
      img.onerror = () => resolve('');
      img.src = dataUrl;
    };
    reader.onerror = () => resolve('');
    reader.readAsDataURL(file);
  });
}

function SidebarCard({ title, icon, children }: { title: string; icon: string; children: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-white/5 bg-surface/50 p-5 backdrop-blur-xl">
      <div className="flex items-center gap-2.5 mb-4">
        <span className="material-icons-outlined text-primary text-xl">{icon}</span>
        <h3 className="text-sm font-semibold text-white">{title}</h3>
      </div>
      {children}
    </div>
  );
}

export default function Analyze() {
  const [step, setStep] = useState(1);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [segmentationMode, setSegmentationMode] = useState<SegmentationMode>('auto');
  const { result, isLoading, error, analyze, reset } = useAnalysis();
  const { token, isAuthenticated } = useAuth();
  const [guestUsage, setGuestUsage] = useState<GuestUsage>(() => getGuestUsage());

  useEffect(() => {
    if (isAuthenticated) return;
    setGuestUsage(getGuestUsage());
  }, [isAuthenticated]);

  const handleImageUpload = useCallback(
    async (file: File) => {
      if (!isAuthenticated) {
        if (guestUsage.remaining <= 0) {
          window.dispatchEvent(new Event('hemavision:open-auth-modal'));
          return;
        }
      }

      setImageFile(file);
      const preview = await fileToPreviewUrl(file);
      setImagePreview(preview || null);
      setStep(2);
      await analyze(file, segmentationMode, token);
      if (!isAuthenticated) {
        setGuestUsage(incrementGuestUsage());
      }
      setStep(3);
    },
    [analyze, segmentationMode, token, isAuthenticated, guestUsage.remaining]
  );

  const handleReset = useCallback(() => {
    setStep(1);
    setImageFile(null);
    setImagePreview(null);
    reset();
  }, [reset]);

  const fileExtIcon = imageFile?.name?.match(/\.(tiff?|bmp)$/i) ? 'description' : 'image';
  const isGuestBlocked = !isAuthenticated && guestUsage.remaining <= 0;

  return (
    <div className="min-h-screen py-8 lg:py-12">
      <div className="section-container">
        
        {/* Page Header */}
        <header className="mb-8">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-6">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-2xl flex items-center justify-center bg-primary/10 border border-primary/20 shadow-[0_0_20px_rgba(19,127,236,0.15)]">
                <span className="material-icons-outlined text-primary text-2xl">biotech</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white tracking-tight">Diagnostic Session</h1>
                <p className="text-sm text-slate-400 mt-0.5">Automated AML screening and morphological analysis</p>
              </div>
            </div>

            {/* Stepper */}
            <div className="flex items-center gap-1.5 p-1.5 rounded-2xl bg-surface/50 border border-white/5 backdrop-blur-xl">
              {stepLabels.map((label, idx) => {
                const stepNum = idx + 1;
                const isDone = step > stepNum;
                const isCurrent = step === stepNum;
                return (
                  <div key={label} className="flex items-center">
                    <div
                      className={clsx(
                        'flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-semibold transition ease-out-custom duration-300',
                        isCurrent ? 'bg-primary text-white shadow-[0_0_12px_rgba(19,127,236,0.4)]' : 
                        isDone ? 'bg-emerald-500/10 text-emerald-400' : 'text-slate-500'
                      )}
                    >
                      <div
                        className={clsx(
                          'w-5 h-5 rounded-full text-[10px] font-bold flex items-center justify-center transition-colors ease-out-custom ease-out-custom',
                          isCurrent ? 'bg-white/20 text-white' : 
                          isDone ? 'bg-emerald-400/20 text-emerald-400' : 'bg-white/5 text-slate-500'
                        )}
                      >
                        {isDone ? '✓' : stepNum}
                      </div>
                      <span className={clsx("hidden sm:block", !isCurrent && !isDone && "opacity-70")}>{label}</span>
                    </div>
                    {idx < stepLabels.length - 1 && (
                      <div className="w-4 h-px mx-1 bg-white/5" />
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </header>

        <div className="grid lg:grid-cols-[1fr_320px] gap-6 items-start">
          
          {/* ── Main Content Area ────────────────────────── */}
          <main className="min-h-[500px]">
            <AnimatePresence mode="wait">
              {step === 1 && (
                <motion.div key="upload" initial={{ opacity: 0, y: 12, scale: 0.95 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.25 }}>
                  {isGuestBlocked ? (
                    <div className="rounded-2xl border border-red-500/20 bg-red-500/5 p-10 text-center backdrop-blur-xl relative overflow-hidden">
                      <div className="w-14 h-14 rounded-2xl mx-auto mb-4 flex items-center justify-center bg-red-500/10 border border-red-500/20 shadow-[0_0_20px_rgba(239,68,68,0.2)]">
                        <span className="material-icons-outlined text-red-400 text-2xl">lock</span>
                      </div>
                      <h3 className="text-xl font-bold text-white mb-2">Analysis Limit Reached</h3>
                      <p className="text-sm text-slate-400 max-w-sm mx-auto mb-6 leading-relaxed">
                        You've used all your guest analyses. Create a free account to unlock unlimited diagnostic sessions, complete case histories, and advanced Grad-CAM tooling.
                      </p>
                      <button
                        onClick={() => window.dispatchEvent(new Event('hemavision:open-auth-modal'))}
                        className="btn-primary !px-8 hover:shadow-[0_0_20px_rgba(19,127,236,0.3)]"
                      >
                        Create Free Account
                      </button>
                    </div>
                  ) : (
                    <UploadZone onUpload={handleImageUpload} />
                  )}
                </motion.div>
              )}

              {step >= 2 && (
                <motion.div key="results" initial={{ opacity: 0, y: 12, scale: 0.95 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.25 }} className="space-y-6">
                  
                  {/* Active File Banner */}
                  {imageFile && (
                    <div className="flex items-center gap-4 p-4 rounded-2xl bg-surface/80 border border-white/5 backdrop-blur-xl">
                      {imagePreview ? (
                        <img src={imagePreview} alt="Upload preview" className="w-14 h-14 rounded-xl object-cover border border-white/10 shadow-sm" />
                      ) : (
                        <div className="w-14 h-14 rounded-xl flex items-center justify-center bg-primary/10 border border-primary/20">
                          <span className="material-icons-outlined text-primary text-xl">{fileExtIcon}</span>
                        </div>
                      )}
                      
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-semibold text-white truncate mb-0.5">{imageFile.name}</p>
                        <p className="text-xs text-slate-400">{(imageFile.size / 1024).toFixed(1)} KB • {imageFile.type || 'Image'}</p>
                      </div>

                      {result && (
                        <div className="flex flex-col items-end gap-1.5">
                          <div className={clsx(
                            'text-xs font-bold px-3 py-1.5 rounded-lg border whitespace-nowrap',
                            result.blast_count === 0 ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' :
                            result.overall_risk_level === 'HIGH RISK' ? 'bg-red-500/10 text-red-400 border-red-500/20' :
                            'bg-amber-500/10 text-amber-400 border-amber-500/20'
                          )}>
                            {result.blast_count === 0 ? 'Normal' : result.overall_risk_level === 'HIGH RISK' ? 'AML Detected' : 'Atypical'}
                          </div>
                          <div className="text-[10px] text-slate-500 font-medium">
                            <span className="uppercase text-slate-300">{result.segmentation_mode_used}</span> Mode
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Processing State */}
                  {isLoading && (
                    <div className="flex flex-col items-center justify-center py-24 rounded-2xl bg-surface/30 border border-white/5 backdrop-blur-xl">
                      <div className="relative mb-6">
                        <div className="absolute inset-0 rounded-full bg-primary/20 animate-ping" style={{ animationDuration: '2.5s' }} />
                        <div className="relative w-16 h-16 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center shadow-[0_0_30px_rgba(19,127,236,0.2)]">
                          <Loader2 className="w-8 h-8 text-primary animate-spin" strokeWidth={2.5} />
                        </div>
                      </div>
                      <h3 className="text-lg font-bold text-white mb-2">Analyzing Morphology...</h3>
                      <p className="text-sm text-slate-400 max-w-sm text-center">
                        Our AI pipeline is processing the slide through Cell Segmentation, ResNet-50 Feature Extraction, and Grad-CAM Generation.
                      </p>
                    </div>
                  )}

                  {/* Error State */}
                  {error && (
                    <div className="text-center py-16 rounded-2xl bg-surface/30 border border-white/5 backdrop-blur-xl">
                      <div className="w-16 h-16 rounded-2xl mx-auto mb-4 flex items-center justify-center bg-red-500/10 border border-red-500/20">
                        <ShieldAlert className="w-8 h-8 text-red-400" />
                      </div>
                      <p className="text-lg text-white font-bold mb-2">Processing Failed</p>
                      <p className="text-sm text-slate-400 mb-6 max-w-md mx-auto">{error}</p>
                      <button onClick={handleReset} className="btn-primary active:scale-[0.97]">
                        Try Another Image
                      </button>
                    </div>
                  )}

                  {/* Results State */}
                  {result && (
                    <div className="space-y-6">
                      <ResultCard result={result} />

                      {result.annotated_image_base64 && result.is_multi_cell && (
                        <div className="rounded-2xl border border-white/5 bg-surface/50 p-6 backdrop-blur-xl">
                          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
                            <div className="flex items-center gap-3">
                              <div className="w-10 h-10 rounded-xl bg-violet-500/10 border border-violet-500/20 flex items-center justify-center text-violet-400">
                                <span className="material-icons-outlined text-lg">grid_on</span>
                              </div>
                              <div>
                                <h3 className="text-base font-bold text-white">Full Slide Segmentation</h3>
                                <p className="text-xs text-slate-400 mt-0.5">{result.num_cells} distinct cells isolated</p>
                              </div>
                            </div>
                          </div>
                          <div className="bg-black/40 rounded-xl overflow-hidden border border-white/5 flex justify-center p-2">
                            <img src={`data:image/png;base64,${result.annotated_image_base64}`} alt="Segmented map" className="max-h-[500px] w-auto object-contain rounded-lg" />
                          </div>
                        </div>
                      )}

                      {result.is_multi_cell && result.cells.some(c => c.gradcam_base64) && (
                        <div className="rounded-2xl border border-white/5 bg-surface/50 p-6 backdrop-blur-xl">
                          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
                            <div className="flex items-center gap-3">
                              <div className="w-10 h-10 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center text-primary shadow-[0_0_15px_rgba(19,127,236,0.15)]">
                                <span className="material-icons-outlined text-lg">visibility</span>
                              </div>
                              <div>
                                <h3 className="text-base font-bold text-white">Isolated Cell Features (Grad-CAM)</h3>
                                <p className="text-xs text-slate-400 mt-0.5">Neural network activation patterns per crop</p>
                              </div>
                            </div>
                            <div className="flex items-center gap-2 text-xs">
                              <span className="text-slate-500">Low</span>
                              <div className="w-24 h-1.5 rounded-full bg-gradient-to-r from-blue-500 via-emerald-400 via-amber-400 to-red-500" />
                              <span className="text-slate-500">High</span>
                            </div>
                          </div>
                          
                          <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-4">
                            {result.cells.map((cell) => {
                              if (!cell.gradcam_base64) return null;
                              const isBlast = cell.prediction.includes('Blast');
                              return (
                                <div key={cell.cell_index} className="group relative rounded-xl border border-white/5 bg-black/20 p-2 transition ease-out-custom hover:bg-white/[0.02]">
                                  <div className="relative rounded-lg overflow-hidden border border-white/5 mb-3 bg-black/40 aspect-square flex items-center justify-center">
                                    <img src={`data:image/png;base64,${cell.gradcam_base64}`} alt={`Cell ${cell.cell_index}`} className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110" />
                                    <div className="absolute top-2 left-2 px-2 py-0.5 rounded text-[10px] font-bold bg-black/60 text-white backdrop-blur-md border border-white/10">
                                      #{cell.cell_index}
                                    </div>
                                  </div>
                                  <div className="flex items-center justify-between px-1">
                                    <span className={clsx('text-xs font-bold', isBlast ? 'text-red-400' : 'text-emerald-400')}>
                                      {isBlast ? 'Blast' : 'Normal'}
                                    </span>
                                    <span className="text-xs text-slate-400">
                                      {(cell.confidence * 100).toFixed(1)}%
                                    </span>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}

                      {!result.is_multi_cell && (
                        <div className="rounded-2xl border border-white/5 bg-surface/50 overflow-hidden backdrop-blur-xl p-1">
                          <GradCAMViewer
                            originalImage={
                              (result.cells[0]?.cell_image_base64
                                ? `data:image/png;base64,${result.cells[0].cell_image_base64}`
                                : imagePreview)
                            }
                            gradcamHeatmapBase64={result.cells[0]?.gradcam_heatmap_base64 ?? null}
                            gradcamBase64={result.cells[0]?.gradcam_base64 ?? null}
                          />
                        </div>
                      )}
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </main>

          {/* ── Sidebar ──────────────────────────────────── */}
          <aside className="space-y-4">
            
            {/* Actions (visible when session active) */}
            <AnimatePresence>
              {step > 1 && (
                <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}>
                  <button
                    onClick={handleReset}
                    className="w-full mb-4 flex items-center justify-center gap-2 p-3.5 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 text-white font-semibold transition ease-out-custom duration-300 active:scale-[0.97]"
                  >
                    <RotateCcw size={16} />
                    New Diagnostic Session
                  </button>
                </motion.div>
              )}
            </AnimatePresence>

            <SidebarCard title="Processing Settings" icon="tune">
              <div className="space-y-3">
                <p className="text-xs text-slate-400 leading-relaxed mb-4">
                  Adjust how the AI handles the clinical image. Auto detects the best mode.
                </p>
                
                {(['auto', 'single', 'multi'] as SegmentationMode[]).map((mode) => {
                  const active = segmentationMode === mode;
                  return (
                    <button
                      key={mode}
                      onClick={() => setSegmentationMode(mode)}
                      className={clsx(
                        'w-full flex items-center justify-between p-3 rounded-xl border text-sm transition ease-out-custom duration-300',
                        active 
                          ? 'border-primary/40 bg-primary/10 text-primary shadow-[0_0_15px_rgba(19,127,236,0.1)]' 
                          : 'border-white/5 bg-black/20 text-slate-400 hover:bg-white/5 hover:text-slate-300'
                      )}
                      disabled={step > 1}
                    >
                      <span className="font-semibold capitalize">{mode}</span>
                      {active && <span className="material-icons-outlined text-[16px]">check_circle</span>}
                    </button>
                  );
                })}
              </div>
            </SidebarCard>

            {!isAuthenticated && (
              <SidebarCard title="Guest Usage" icon="account_circle">
                <div className="mb-4">
                  <div className="flex items-center justify-between text-sm mb-2">
                    <span className="text-slate-400">Analyses used</span>
                    <span className="text-white font-bold border border-white/10 bg-black/30 px-2 py-0.5 rounded">
                      {guestUsage.used} / {guestUsage.limit}
                    </span>
                  </div>
                  <div className="h-2 w-full bg-black/40 rounded-full overflow-hidden border border-white/5">
                    <div 
                      className={clsx("h-full transition ease-out-custom duration-500", guestUsage.remaining === 0 ? "bg-red-500" : "bg-primary")}
                      style={{ width: `${(guestUsage.used / guestUsage.limit) * 100}%` }}
                    />
                  </div>
                  <p className="text-[11px] text-slate-500 mt-3 leading-relaxed">
                    {guestUsage.remaining === 0 
                      ? "Limit reached. Sign in to continue."
                      : `${guestUsage.remaining} analyses left without an account.`}
                  </p>
                </div>
                
                <button
                  onClick={() => window.dispatchEvent(new Event('hemavision:open-auth-modal'))}
                  className="w-full py-2.5 rounded-lg border border-primary/40 text-primary text-xs font-bold hover:bg-primary/10 transition-colors ease-out-custom ease-out-custom"
                >
                  Create Account
                </button>
              </SidebarCard>
            )}

            <SidebarCard title="AI Confidence" icon="verified">
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
                    <span className="material-icons-outlined text-sm">precision_manufacturing</span>
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-white">96.8% Accuracy</h4>
                    <p className="text-[10px] text-slate-400 mt-0.5">ResNet-50 dual-stream</p>
                  </div>
                </div>
                <p className="text-[10px] text-slate-500 leading-relaxed border-t border-white/5 pt-3">
                  Predictions are probabilistic and intended for research/clinical support, not a definitive diagnosis.
                </p>
              </div>
            </SidebarCard>

          </aside>
        </div>
      </div>
    </div>
  );
}

import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';
import ResultCard from "../components/ResultCard";
import GradCAMViewer from "../components/GradCAMViewer";
import {
  getAnalyses,
  getAnalysisStats,
  deleteAnalysis,
  type AnalysisRecord,
  type AnalysisStats,
} from '../lib/api';
import { useAuth } from '../contexts/AuthContext';

export default function History() {
  const { token, isAuthenticated, loading: authLoading } = useAuth();
  const [records, setRecords] = useState<AnalysisRecord[]>([]);
  const [stats, setStats] = useState<AnalysisStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    if (authLoading) return;
    if (!isAuthenticated || !token) {
      setRecords([]);
      setStats(null);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const [analysesRes, statsRes] = await Promise.all([
        getAnalyses(50, 0, token),
        getAnalysisStats(token),
      ]);
      setRecords(analysesRes.records);
      setStats(statsRes);
    } catch (err: any) {
      if (err.response?.status === 401) {
        setError('Your session has expired. Please sign in again.');
      } else {
        setError('Could not load history. Make sure the backend is running.');
      }
    } finally {
      setLoading(false);
    }
  }, [authLoading, isAuthenticated, token]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleDelete = async (id: string) => {
    try {
      await deleteAnalysis(id, token);
      setRecords((prev) => prev.filter((r) => r.id !== id));
      const statsRes = await getAnalysisStats(token);
      setStats(statsRes);
    } catch {
      // Silently fail
    }
  };

  const formatDate = (iso: string) => {
    const d = new Date(iso);
    return d.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const riskBadge = (level: string) => {
    const cls =
      level === 'HIGH RISK'
        ? 'badge-danger'
        : level === 'MODERATE RISK'
        ? 'badge-warning'
        : 'badge-success';
    return <span className={cls}>{level}</span>;
  };

  // ── Unauthenticated state ───────────────────────────
  if (!authLoading && !isAuthenticated) {
    return (
      <div className="min-h-screen">
        <div className="section-container pt-16 pb-12">
          <h1 className="text-xl font-semibold text-white">History</h1>
          <p className="text-sm text-slate-400 mt-1">
            Review past analyses and compare diagnostic results.
          </p>
        </div>
        <div className="section-container pb-20">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-center py-24 border border-white/5 rounded-2xl bg-surface/50 backdrop-blur-xl relative overflow-hidden"
          >
            {/* Subtle glow behind icon */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-32 h-32 bg-primary/20 blur-[50px] rounded-full" />
            
            <div className="relative">
              <div className="w-16 h-16 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center mx-auto mb-6 shadow-[0_0_20px_rgba(19,127,236,0.15)]">
                <span className="material-icons-outlined text-3xl text-primary">lock</span>
              </div>
              <h3 className="text-xl font-bold text-white mb-2">Sign in to view history</h3>
              <p className="text-sm text-slate-400 max-w-sm mx-auto mb-8 leading-relaxed">
                Create a free account to unlock unlimited diagnostic sessions, save your complete case history, and access advanced Grad-CAM tooling.
              </p>
              <button
                onClick={() => window.dispatchEvent(new Event('hemavision:open-auth-modal'))}
                className="btn-primary !px-8 shadow-[0_0_20px_rgba(19,127,236,0.3)] hover:shadow-[0_0_30px_rgba(19,127,236,0.5)] active:scale-[0.97] transition-transform"
              >
                Sign In or Register
              </button>
            </div>
          </motion.div>
        </div>
      </div>
    );
  }

  // ── Empty state ───────────────────────────
  if (!loading && records.length === 0 && !error) {
    return (
      <div className="min-h-screen">
        <div className="section-container pt-16 pb-12">
          <h1 className="text-xl font-semibold text-white">History</h1>
          <p className="text-sm text-slate-400 mt-1">
            Review past analyses and compare diagnostic results.
          </p>
        </div>
        <div className="section-container pb-20">
          <motion.div
            initial={{ opacity: 0, y: 16, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            className="text-center py-24 border border-slate-800 rounded-xl bg-surface"
          >
            <span className="material-icons-outlined text-5xl text-slate-700 mb-4 block">science</span>
            <h3 className="text-base font-semibold text-white mb-2">No analyses yet</h3>
            <p className="text-sm text-slate-400 max-w-[360px] mx-auto mb-8">
              Your analysis history will appear here once you run your first diagnosis.
            </p>
            <a href="/analyze" className="btn-primary">
              Start First Analysis
            </a>
          </motion.div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen">
      {/* ── Header ─────────────────────────────── */}
      <div className="border-b border-slate-800">
        <div className="section-container py-8 flex items-end justify-between">
          <div>
            <h1 className="text-xl font-semibold text-white">History</h1>
            <p className="text-sm text-slate-400 mt-1">
              {stats ? `${stats.total_analyses} total analyses` : 'Loading...'}
            </p>
          </div>
          <button
            onClick={fetchData}
            className="btn-ghost flex items-center gap-1.5 active:scale-[0.97]"
          >
            <span className={clsx('material-icons-outlined text-base', loading && 'animate-spin')}>refresh</span>
            Refresh
          </button>
        </div>
      </div>

      <div className="section-container py-10">
        {/* ── Stats ──────────────────────────────── */}
        {stats && stats.total_analyses > 0 && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-10">
            <StatCard label="Total" value={stats.total_analyses} icon="analytics" />
            <StatCard label="AML Detected" value={stats.aml_detected} icon="warning" accent />
            <StatCard label="Normal" value={stats.normal_detected} icon="check_circle" />
            <StatCard label="Avg Confidence" value={`${(stats.avg_confidence * 100).toFixed(1)}%`} icon="speed" />
          </div>
        )}

        {/* ── Error ──────────────────────────────── */}
        {error && (
          <div className="mb-8 p-4 rounded-xl border border-red-500/20 bg-red-500/5">
            <p className="text-sm text-slate-300">{error}</p>
          </div>
        )}

        {/* ── Loading skeleton ───────────────────── */}
        {loading && (
          <div className="space-y-3">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="animate-pulse rounded-xl border border-slate-800 bg-surface p-5">
                <div className="flex items-center gap-4">
                  <div className="w-8 h-8 bg-slate-700 rounded-full" />
                  <div className="flex-1 space-y-2">
                    <div className="h-3.5 bg-slate-700 rounded w-1/4" />
                    <div className="h-3 bg-slate-700 rounded w-1/3" />
                  </div>
                  <div className="h-6 w-20 bg-slate-700 rounded-full" />
                </div>
              </div>
            ))}
          </div>
        )}

        {/* ── Records ────────────────────────────── */}
        {!loading && records.length > 0 && (
          <div className="space-y-2">
            <AnimatePresence initial={false}>
              {records.map((record, idx) => {
                const isExpanded = expandedId === record.id;
                const isAML = record.prediction.includes('AML');
                return (
                  <motion.div
                    key={record.id}
                    initial={{ opacity: 0, y: 8, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ delay: idx * 0.015 }}
                    className="rounded-xl border border-slate-800 bg-surface overflow-hidden"
                  >
                    {/* Row */}
                    <button
                      onClick={() => setExpandedId(isExpanded ? null : record.id)}
                      className="w-full flex items-center gap-4 px-5 py-4 text-left hover:bg-white/[0.02] transition-colors ease-out-custom ease-out-custom"
                    >
                      {/* Status indicator */}
                      <div
                        className={clsx(
                          'w-2 h-2 rounded-full flex-shrink-0',
                          isAML ? 'bg-red-500' : 'bg-emerald-500'
                        )}
                      />

                      {/* Info */}
                      <div className="flex-1 min-w-0">
                        <span className="text-sm font-semibold text-white truncate block">
                          {record.prediction}
                        </span>
                        <span className="text-xs text-slate-500 mt-0.5 block">
                          {formatDate(record.created_at)}
                          {record.image_filename && ` \u00B7 ${record.image_filename}`}
                        </span>
                      </div>

                      {/* Confidence + Risk */}
                      <div className="hidden sm:flex items-center gap-3">
                        <span className="text-sm font-medium text-slate-300 tabular-nums">
                          {(record.confidence * 100).toFixed(1)}%
                        </span>
                        {riskBadge(record.risk_level)}
                      </div>

                      {/* Expand */}
                      <span className="material-icons-outlined text-slate-500 text-lg">
                        {isExpanded ? 'expand_less' : 'expand_more'}
                      </span>
                    </button>

                    {/* Expanded detail */}
                    <AnimatePresence>
                      {isExpanded && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: 'auto', opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          transition={{ duration: 0.2 }}
                          className="overflow-hidden"
                        >
                          <div className="px-5 pb-5 pt-4 border-t border-slate-800">
                            {(() => {
                              let cells: any[] = [];
                              try {
                                if (record.cells_data) {
                                  const parsed = JSON.parse(record.cells_data);
                                  if (Array.isArray(parsed)) cells = parsed;
                                }
                              } catch(e) {}
                              const isMultiCell = cells.length > 1;

                              const confidenceValue = record.confidence != null ? record.confidence : record.probability;
                              const confidence = confidenceValue || 0;
                              const isAmlOverall = record.prediction?.includes('AML') || record.risk_level === 'HIGH RISK';

                              const mockResult: any = {
                                is_multi_cell: isMultiCell,
                                num_cells: cells.length || 1,
                                estimated_total_cells: cells.length || 1,
                                segmentation_mode_used: isMultiCell ? 'multi' : 'single',
                                overall_prediction: record.prediction,
                                overall_risk_level: record.risk_level,
                                overall_risk_color: record.risk_color,
                                blast_count: isMultiCell ? cells.filter(c => c.prediction?.includes('AML') || c.prediction?.includes('Blast')).length : (isAmlOverall ? 1 : 0),
                                normal_count: isMultiCell ? cells.filter(c => !c.prediction?.includes('AML') && !c.prediction?.includes('Blast')).length : (isAmlOverall ? 0 : 1),
                                blast_percentage:
                                  isMultiCell && cells.length > 0
                                    ? (cells.filter(c => c.prediction?.includes('AML') || c.prediction?.includes('Blast')).length / cells.length) * 100
                                    : confidence * 100,
                                cells: cells.length > 0 ? cells : [{
                                  cell_index: 1,
                                  prediction: record.prediction,
                                  probability: record.probability,
                                  confidence: record.confidence,
                                  risk_level: record.risk_level,
                                  risk_color: record.risk_color,
                                  gradcam_base64: record.gradcam_base64,
                                  cell_image_base64: record.source_image_base64
                                }],
                                annotated_image_base64: isMultiCell ? record.gradcam_base64 : null,
                                inference_time_ms: record.inference_time_ms,
                                segmentation_message: ""
                              };

                              return (
                                <div className="space-y-6">
                                  <ResultCard result={mockResult} />

                                  {mockResult.annotated_image_base64 && isMultiCell && (
                                    <div className="rounded-2xl border border-white/5 bg-surface/50 p-6 backdrop-blur-xl">
                                      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
                                        <div className="flex items-center gap-3">
                                          <div className="w-10 h-10 rounded-xl bg-violet-500/10 border border-violet-500/20 flex items-center justify-center text-violet-400">
                                            <span className="material-icons-outlined text-lg">grid_on</span>
                                          </div>
                                          <div>
                                            <h3 className="text-base font-bold text-white">Full Slide Segmentation</h3>
                                            <p className="text-xs text-slate-400 mt-0.5">{mockResult.num_cells} distinct cells isolated</p>
                                          </div>
                                        </div>
                                      </div>
                                      <div className="bg-black/40 rounded-xl overflow-hidden border border-white/5 flex justify-center p-2">
                                        <img src={`data:image/png;base64,${mockResult.annotated_image_base64}`} alt="Segmented map" className="max-h-[500px] w-auto object-contain rounded-lg" />
                                      </div>
                                    </div>
                                  )}

                                  {isMultiCell && mockResult.cells.some((c: any) => c.gradcam_base64) && (
                                    <div className="rounded-2xl border border-white/5 bg-surface/50 p-6 backdrop-blur-xl">
                                      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
                                        <div className="flex items-center gap-3">
                                          <div className="w-10 h-10 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center text-primary shadow-[0_0_15px_rgba(19,127,236,0.15)]">
                                            <span className="material-icons-outlined text-lg">visibility</span>
                                          </div>
                                          <div>
                                            <h3 className="text-base font-bold text-white">Isolated Cell Features (Grad-CAM)</h3>
                                            <p className="text-xs text-slate-400 mt-0.5">Neural network activation patterns per crop ({cells.length} cells)</p>
                                          </div>
                                        </div>
                                        <div className="flex items-center gap-2 text-xs">
                                          <span className="text-slate-500">Low</span>
                                          <div className="w-24 h-1.5 rounded-full bg-gradient-to-r from-blue-500 via-emerald-400 via-amber-400 to-red-500" />
                                          <span className="text-slate-500">High</span>
                                        </div>
                                      </div>
                                      
                                      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-4">
                                        {mockResult.cells.map((cell: any, idx: number) => {
                                          const isCellAml = cell.prediction?.includes('AML') || cell.prediction?.includes('Blast');
                                          const imgBase64 = cell.cell_image_base64;
                                          const gradcamSrc = cell.gradcam_base64 ? `data:image/jpeg;base64,${cell.gradcam_base64}` : null;
                                          const confVal = cell.confidence != null ? cell.confidence : cell.probability;
                                          
                                          return (
                                            <div key={idx} className="group relative rounded-xl border border-white/5 bg-black/20 p-2 transition ease-out-custom hover:bg-white/[0.02]">
                                              <div className="aspect-square relative overflow-hidden bg-black/40 rounded-lg mb-3 flex items-center justify-center">
                                                {imgBase64 ? (
                                                  <>
                                                    <img
                                                      src={`data:image/jpeg;base64,${imgBase64}`}
                                                      alt={`Cell ${idx}`}
                                                      className="absolute inset-0 w-full h-full object-cover group-hover:opacity-0 transition-opacity duration-500"
                                                    />
                                                    {gradcamSrc && (
                                                      <img
                                                        src={gradcamSrc}
                                                        alt={`GradCAM ${idx}`}
                                                        className="absolute inset-0 w-full h-full object-cover opacity-0 group-hover:opacity-100 transition-opacity duration-500 scale-105"
                                                      />
                                                    )}
                                                  </>
                                                ) : gradcamSrc ? (
                                                   <img
                                                      src={gradcamSrc}
                                                      alt={`GradCAM ${idx}`}
                                                      className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
                                                   />
                                                ) : (
                                                  <div className="w-full h-full flex items-center justify-center opacity-20">
                                                    <span className="material-icons-outlined text-2xl text-white">image</span>
                                                  </div>
                                                )}
                                                
                                                <div className="absolute top-2 left-2 px-2 py-0.5 rounded text-[10px] font-bold bg-black/60 text-white backdrop-blur-md border border-white/10">
                                                  #{cell.cell_index !== undefined ? cell.cell_index : idx + 1}
                                                </div>

                                                {gradcamSrc && imgBase64 && (
                                                  <div className="absolute bottom-2 inset-x-0 text-center text-[10px] uppercase font-bold text-white/70 opacity-100 group-hover:opacity-0 transition-opacity pointer-events-none drop-shadow-md">
                                                    Hover for Grad-CAM
                                                  </div>
                                                )}
                                              </div>

                                              <div className="flex items-center justify-between px-1">
                                                <span className={clsx('text-xs font-bold', isCellAml ? 'text-red-400' : 'text-emerald-400')}>
                                                  {isCellAml ? 'Blast' : 'Normal'}
                                                </span>
                                                <span className="text-xs text-slate-400">
                                                  {confVal ? (confVal * 100).toFixed(1) : 0}%
                                                </span>
                                              </div>
                                            </div>
                                          );
                                        })}
                                      </div>
                                    </div>
                                  )}

                                  {!isMultiCell && (
                                    <div className="rounded-2xl border border-white/5 bg-surface/50 overflow-hidden backdrop-blur-xl p-1">
                                      <GradCAMViewer
                                        originalImage={
                                          record.source_image_base64
                                            ? `data:image/png;base64,${record.source_image_base64}`
                                            : (mockResult.cells[0]?.cell_image_base64 ? `data:image/png;base64,${mockResult.cells[0].cell_image_base64}` : null)
                                        }
                                        gradcamHeatmapBase64={mockResult.cells[0]?.gradcam_heatmap_base64 ?? null}
                                        gradcamBase64={mockResult.cells[0]?.gradcam_base64 ?? null}
                                      />
                                    </div>
                                  )}

                                </div>
                              );
                            })()}

                            {/* Actions */}
                            <div className="flex justify-end mt-4 pt-4 border-t border-white/5">
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDelete(record.id);
                                }}
                                className="flex items-center gap-1.5 text-xs text-slate-500 
                                           hover:text-red-400 transition-colors ease-out-custom ease-out-custom px-3 py-1.5 
                                           rounded-lg hover:bg-red-500/5"
                              >
                                <span className="material-icons-outlined text-sm">delete</span>
                                Delete Record
                              </button>
                            </div>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </motion.div>
                );
              })}
            </AnimatePresence>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Sub-components ──────────────────────────────────

function StatCard({
  label,
  value,
  icon,
  accent,
}: {
  label: string;
  value: string | number;
  icon: string;
  accent?: boolean;
}) {
  return (
    <div className="py-5 px-5 rounded-xl border border-slate-800 bg-surface">
      <div className="flex items-center gap-2 mb-2">
        <span className={clsx('material-icons-outlined text-lg', accent ? 'text-red-400' : 'text-primary')}>{icon}</span>
        <span className="text-xs text-slate-500 uppercase tracking-wider">{label}</span>
      </div>
      <div className={clsx(
        'text-2xl font-bold tracking-tight',
        accent ? 'text-red-400' : 'text-white'
      )}>
        {value}
      </div>
    </div>
  );
}


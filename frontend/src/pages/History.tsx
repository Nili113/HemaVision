import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';
import {
  getAnalyses,
  getAnalysisStats,
  deleteAnalysis,
  type AnalysisRecord,
  type AnalysisStats,
} from '../lib/api';

export default function History() {
  const [records, setRecords] = useState<AnalysisRecord[]>([]);
  const [stats, setStats] = useState<AnalysisStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [analysesRes, statsRes] = await Promise.all([
        getAnalyses(50, 0),
        getAnalysisStats(),
      ]);
      setRecords(analysesRes.records);
      setStats(statsRes);
    } catch {
      setError('Could not load history. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleDelete = async (id: string) => {
    try {
      await deleteAnalysis(id);
      setRecords((prev) => prev.filter((r) => r.id !== id));
      const statsRes = await getAnalysisStats();
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
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
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
            className="btn-ghost flex items-center gap-1.5"
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
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ delay: idx * 0.015 }}
                    className="rounded-xl border border-slate-800 bg-surface overflow-hidden"
                  >
                    {/* Row */}
                    <button
                      onClick={() => setExpandedId(isExpanded ? null : record.id)}
                      className="w-full flex items-center gap-4 px-5 py-4 text-left hover:bg-white/[0.02] transition-colors"
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
                          <div className="px-5 pb-5 pt-1 border-t border-slate-800">
                            {/* Mobile risk badge */}
                            <div className="sm:hidden mb-4 mt-3">
                              {riskBadge(record.risk_level)}
                            </div>

                            {/* Detail grid */}
                            <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm mb-4 mt-3">
                              <DetailItem label="Probability" value={`${(record.probability * 100).toFixed(2)}%`} />
                              <DetailItem label="Confidence" value={`${(record.confidence * 100).toFixed(2)}%`} />
                              <DetailItem label="Inference" value={`${record.inference_time_ms.toFixed(1)}ms`} />
                            </div>

                            {/* Analysis method */}
                            <div className="flex flex-wrap gap-2 mb-4">
                              <span className="px-2.5 py-1 rounded-md text-xs font-medium border bg-primary/10 text-primary border-primary/20">
                                Multimodal: CNN + Morphology
                              </span>
                            </div>

                            {/* Actions */}
                            <div className="flex justify-end">
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDelete(record.id);
                                }}
                                className="flex items-center gap-1.5 text-xs text-slate-500 
                                           hover:text-red-400 transition-colors px-3 py-1.5 
                                           rounded-lg hover:bg-red-500/5"
                              >
                                <span className="material-icons-outlined text-sm">delete</span>
                                Delete
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

function DetailItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-slate-500 uppercase tracking-wider">{label}</div>
      <div className="text-sm font-medium text-white mt-0.5">{value}</div>
    </div>
  );
}

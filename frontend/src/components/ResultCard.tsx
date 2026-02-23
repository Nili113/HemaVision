import { motion } from 'framer-motion';
import clsx from 'clsx';
import type { MultiCellResponse } from '../lib/api';

interface ResultCardProps {
  result: MultiCellResponse | null;
}

export default function ResultCard({ result }: ResultCardProps) {
  if (!result) {
    return (
      <div className="rounded-xl border border-slate-800 bg-surface p-8">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-slate-700 rounded w-1/3" />
          <div className="h-4 bg-slate-700 rounded w-1/2" />
          <div className="h-4 bg-slate-700 rounded w-2/3" />
        </div>
      </div>
    );
  }

  const isNormal = result.blast_count === 0;
  const isHighRisk = result.overall_risk_level === 'HIGH RISK';

  const accentColor = isNormal ? 'emerald' : isHighRisk ? 'red' : 'amber';
  const accentMap = {
    emerald: { text: 'text-emerald-400', bg: 'bg-emerald-500', bgSoft: 'bg-emerald-500/10', border: 'border-emerald-500/20', glow: 'rgba(52,211,153,0.08)' },
    red:     { text: 'text-red-400',     bg: 'bg-red-500',     bgSoft: 'bg-red-500/10',     border: 'border-red-500/20',     glow: 'rgba(239,68,68,0.08)'  },
    amber:   { text: 'text-amber-400',   bg: 'bg-amber-500',   bgSoft: 'bg-amber-500/10',   border: 'border-amber-500/20',   glow: 'rgba(245,158,11,0.08)' },
  };
  const a = accentMap[accentColor];

  const barColor = isNormal ? 'bg-emerald-500' : isHighRisk ? 'bg-red-500' : 'bg-amber-500';

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.25, 0.1, 0.25, 1] }}
    >
      <div
        className={clsx('rounded-xl border bg-surface overflow-hidden', a.border)}
        style={{ boxShadow: `0 0 60px ${a.glow}, 0 1px 3px rgba(0,0,0,0.3)` }}
      >
        {/* ── Hero banner ───────────────────────────── */}
        <div className="relative px-6 pt-6 pb-5" style={{ background: `linear-gradient(135deg, ${a.glow} 0%, transparent 70%)` }}>
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-start gap-4">
              {/* Status icon */}
              <div className={clsx('w-12 h-12 rounded-xl flex items-center justify-center shrink-0', a.bgSoft)}>
                <span className={clsx('material-icons-outlined text-2xl', a.text)}>
                  {isNormal ? 'check_circle' : isHighRisk ? 'warning' : 'info'}
                </span>
              </div>

              <div>
                <h2 className={clsx('text-lg font-bold leading-snug', a.text)}>
                  {result.overall_prediction}
                </h2>
                <div className="flex flex-wrap items-center gap-2 mt-1.5">
                  <span className={clsx(
                    'inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-bold uppercase tracking-wider border',
                    a.bgSoft, a.text, a.border
                  )}>
                    <span className={clsx('w-1.5 h-1.5 rounded-full', a.bg)} />
                    {result.overall_risk_level}
                  </span>
                  <span className="text-xs text-slate-500 tabular-nums">
                    {result.inference_time_ms.toFixed(0)} ms inference
                  </span>
                </div>
              </div>
            </div>

            {/* Large percentage badge */}
            {result.is_multi_cell ? (
              <div className="text-right shrink-0">
                <div className={clsx('text-3xl font-black tabular-nums leading-none', a.text)}>
                  {result.blast_percentage.toFixed(0)}%
                </div>
                <div className="text-[10px] text-slate-500 mt-1 uppercase tracking-wider font-medium">Blast Rate</div>
              </div>
            ) : (
              <div className="text-right shrink-0">
                <div className={clsx('text-3xl font-black tabular-nums leading-none', a.text)}>
                  {((result.cells[0]?.confidence ?? 0) * 100).toFixed(0)}%
                </div>
                <div className="text-[10px] text-slate-500 mt-1 uppercase tracking-wider font-medium">Confidence</div>
              </div>
            )}
          </div>
        </div>

        {/* ── Body ─────────────────────────────────── */}
        <div className="px-6 pb-6">
          {/* Multi-cell info */}
          {result.is_multi_cell && (
            <div className="flex items-center gap-2 text-xs text-primary/80 font-medium py-3 border-b border-slate-800/60 mb-5">
              <span className="material-icons-outlined text-sm">grid_view</span>
              {result.num_cells} cells segmented — {result.blast_count} blast, {result.normal_count} normal
            </div>
          )}

          {/* Metric bars */}
          <div className="space-y-4 mt-4">
            {result.is_multi_cell ? (
              <>
                <MetricBar
                  label="Blast Cells"
                  value={result.blast_percentage / 100}
                  barClass={barColor}
                  suffix={`${result.blast_count} / ${result.num_cells}`}
                />
                <MetricBar
                  label="Normal Cells"
                  value={result.normal_count / result.num_cells}
                  barClass="bg-emerald-500"
                  suffix={`${result.normal_count} / ${result.num_cells}`}
                />
              </>
            ) : (
              <>
                <MetricBar
                  label="Confidence"
                  value={result.cells[0]?.confidence ?? 0}
                  barClass={barColor}
                />
                <MetricBar
                  label="AML Probability"
                  value={result.cells[0]?.probability ?? 0}
                  barClass="bg-gradient-to-r from-emerald-500 via-amber-500 to-red-500"
                  showScale
                />
              </>
            )}
          </div>

          {/* Per-cell breakdown (multi-cell only) */}
          {result.is_multi_cell && result.cells.length > 1 && (
            <div className="mt-6 pt-5 border-t border-slate-800/60">
              <div className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.12em] mb-3">
                Per-Cell Breakdown
              </div>
              <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1">
                {result.cells.map((cell) => {
                  const cellIsBlast = cell.prediction.includes('Blast');
                  return (
                    <div
                      key={cell.cell_index}
                      className="flex items-center justify-between py-2 px-3 rounded-lg hover:bg-slate-800/40 transition-colors"
                    >
                      <div className="flex items-center gap-2.5">
                        <div
                          className={clsx(
                            'w-6 h-6 rounded text-[10px] font-bold flex items-center justify-center',
                            cellIsBlast ? 'bg-red-500/15 text-red-400' : 'bg-emerald-500/15 text-emerald-400'
                          )}
                        >
                          {cell.cell_index}
                        </div>
                        <span className={clsx('text-xs font-semibold', cellIsBlast ? 'text-red-400' : 'text-emerald-400')}>
                          {cellIsBlast ? 'Blast' : 'Normal'}
                        </span>
                      </div>
                      <div className="flex items-center gap-2.5">
                        <div className="w-16 bg-slate-800 rounded-full h-1 overflow-hidden">
                          <div
                            className={clsx('h-full rounded-full transition-all', cellIsBlast ? 'bg-red-500' : 'bg-emerald-500')}
                            style={{ width: `${cell.confidence * 100}%` }}
                          />
                        </div>
                        <span className="text-[11px] text-slate-400 tabular-nums w-10 text-right font-medium">
                          {(cell.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Analysis Method */}
          <div className="mt-6 pt-5 border-t border-slate-800/60">
            <div className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.12em] mb-3">
              Analysis Pipeline
            </div>
            <div className="grid grid-cols-4 gap-2">
              <PipelineStep icon="photo_camera" label="ResNet-50" sub="Visual" />
              <PipelineStep icon="scatter_plot" label="20 Features" sub="Morphology" />
              <PipelineStep icon="merge_type" label="Late 2080-d" sub="Fusion" />
              <PipelineStep icon="apps" label={`${result.num_cells} Cell${result.num_cells > 1 ? 's' : ''}`} sub="Analyzed" />
            </div>
          </div>

          {/* Recommendation */}
          {!isNormal && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="mt-5 p-4 rounded-lg border border-red-500/10"
              style={{ background: 'rgba(239,68,68,0.03)' }}
            >
              <div className="flex items-start gap-2.5">
                <span className="material-icons-outlined text-red-400/60 text-base mt-0.5 shrink-0">clinical_notes</span>
                <p className="text-xs text-slate-400 leading-relaxed">
                  <strong className="font-semibold text-slate-200">Clinical Recommendation — </strong>
                  {result.blast_percentage >= 20
                    ? `${result.blast_count} of ${result.num_cells} cells exhibit blast morphology (${result.blast_percentage.toFixed(1)}%). Immediate referral to hematology is recommended for confirmatory flow cytometry and cytogenetic analysis.`
                    : `Atypical cell morphology detected. Further examination by a qualified hematologist is advised.`}
                </p>
              </div>
            </motion.div>
          )}

          {/* Disclaimer */}
          <p className="text-[10px] text-slate-600 mt-5 leading-relaxed text-center">
            Research tool only — not a substitute for professional medical diagnosis.
          </p>
        </div>
      </div>
    </motion.div>
  );
}

/* ── Sub-components ─────────────────────────── */

function MetricBar({
  label,
  value,
  barClass,
  showScale,
  suffix,
}: {
  label: string;
  value: number;
  barClass: string;
  showScale?: boolean;
  suffix?: string;
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-xs font-medium text-slate-400">{label}</span>
        <span className="text-xs font-bold text-white tabular-nums">
          {suffix || `${(value * 100).toFixed(1)}%`}
        </span>
      </div>
      <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 0.8, ease: 'easeOut', delay: 0.3 }}
          className={clsx('h-full rounded-full', barClass)}
        />
      </div>
      {showScale && (
        <div className="flex justify-between text-[10px] text-slate-600 mt-1">
          <span>Normal</span>
          <span>AML Blast</span>
        </div>
      )}
    </div>
  );
}

function PipelineStep({ icon, label, sub }: { icon: string; label: string; sub: string }) {
  return (
    <div className="flex flex-col items-center p-2.5 rounded-lg bg-slate-800/30 border border-slate-800/50">
      <span className="material-icons-outlined text-primary/60 text-base mb-1">{icon}</span>
      <div className="text-[11px] font-semibold text-white">{label}</div>
      <div className="text-[9px] text-slate-500 uppercase tracking-wider">{sub}</div>
    </div>
  );
}

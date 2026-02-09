import { motion } from 'framer-motion';
import clsx from 'clsx';
import type { PredictionResponse } from '../lib/api';

interface ResultCardProps {
  result: PredictionResponse | null;
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

  const isNormal = result.prediction.includes('Normal');
  const isHighRisk = result.risk_level === 'HIGH RISK';

  const statusColor = isNormal
    ? 'text-emerald-400'
    : isHighRisk
    ? 'text-red-400'
    : 'text-amber-400';

  const barColor = isNormal
    ? 'bg-emerald-500'
    : isHighRisk
    ? 'bg-red-500'
    : 'bg-amber-500';

  const badgeCls = isNormal
    ? 'badge-success'
    : isHighRisk
    ? 'badge-danger'
    : 'badge-warning';

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.25, 0.1, 0.25, 1] }}
    >
      <div className="rounded-xl border border-slate-800 bg-surface p-8">
        {/* Top: Prediction + Risk */}
        <div className="flex items-start justify-between mb-8">
          <div>
            <h2 className={clsx('text-title', statusColor)}>{result.prediction}</h2>
            <div className="flex items-center gap-3 mt-2">
              <span className={badgeCls}>{result.risk_level}</span>
              <span className="text-sm text-slate-500 tabular-nums">
                {result.inference_time_ms.toFixed(1)} ms
              </span>
            </div>
          </div>
        </div>

        {/* Metrics */}
        <div className="space-y-5">
          <MetricBar
            label="Confidence"
            value={result.confidence}
            barClass={barColor}
          />
          <MetricBar
            label="AML Probability"
            value={result.probability}
            barClass="bg-gradient-to-r from-emerald-500 via-amber-500 to-red-500"
            showScale
          />
        </div>

        {/* Patient Context */}
        {result.patient_context && (
          <div className="mt-8 pt-6 border-t border-slate-800">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-4">
              Patient Context
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-4 text-center">
              <ContextItem label="Age" value={`${result.patient_context.age}y`} />
              <ContextItem label="Sex" value={result.patient_context.sex} />
              <ContextItem
                label="NPM1"
                value={result.patient_context.npm1_mutated ? 'Pos' : 'Neg'}
                positive={result.patient_context.npm1_mutated}
              />
              <ContextItem
                label="FLT3"
                value={result.patient_context.flt3_mutated ? 'Pos' : 'Neg'}
                positive={result.patient_context.flt3_mutated}
              />
              <ContextItem
                label="Other"
                value={result.patient_context.genetic_other ? 'Pos' : 'Neg'}
                positive={result.patient_context.genetic_other}
              />
            </div>
          </div>
        )}

        {/* Recommendation */}
        {!isNormal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="mt-6 p-4 rounded-lg bg-red-500/5 border border-red-500/10"
          >
            <p className="text-sm text-slate-300 leading-relaxed">
              <strong className="font-semibold text-white">Recommendation:</strong>{' '}
              This cell shows characteristics consistent with AML blast morphology.
              Further examination by a qualified hematologist is advised.
              Consider confirmatory flow cytometry and cytogenetic analysis.
            </p>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}

function MetricBar({
  label,
  value,
  barClass,
  showScale,
}: {
  label: string;
  value: number;
  barClass: string;
  showScale?: boolean;
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-slate-300">{label}</span>
        <span className="text-sm font-semibold text-white tabular-nums">
          {(value * 100).toFixed(1)}%
        </span>
      </div>
      <div className="w-full bg-slate-700 rounded-full h-1.5 overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 0.8, ease: 'easeOut', delay: 0.3 }}
          className={clsx('h-full rounded-full', barClass)}
        />
      </div>
      {showScale && (
        <div className="flex justify-between text-xs text-slate-500 mt-1">
          <span>Normal</span>
          <span>AML Blast</span>
        </div>
      )}
    </div>
  );
}

function ContextItem({
  label,
  value,
  positive,
}: {
  label: string;
  value: string;
  positive?: boolean;
}) {
  return (
    <div>
      <div className="text-xs text-slate-500 uppercase tracking-wider">{label}</div>
      <div
        className={clsx(
          'text-sm font-semibold mt-0.5',
          positive === true ? 'text-red-400' : positive === false ? 'text-emerald-400' : 'text-white'
        )}
      >
        {value}
      </div>
    </div>
  );
}

import { useState } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft } from 'lucide-react';
import clsx from 'clsx';

interface PatientFormProps {
  onSubmit: (data: {
    age: number;
    sex: string;
    npm1_mutated: boolean;
    flt3_mutated: boolean;
    genetic_other: boolean;
  }) => void;
  onBack: () => void;
}

export default function PatientForm({ onSubmit, onBack }: PatientFormProps) {
  const [age, setAge] = useState(60);
  const [sex, setSex] = useState('Male');
  const [npm1, setNpm1] = useState(false);
  const [flt3, setFlt3] = useState(false);
  const [geneticOther, setGeneticOther] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({
      age,
      sex,
      npm1_mutated: npm1,
      flt3_mutated: flt3,
      genetic_other: geneticOther,
    });
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-surface rounded-xl border border-slate-800 p-6 shadow-card"
    >
      <div className="flex items-center gap-2 mb-6">
        <span className="material-icons-outlined text-primary">description</span>
        <h2 className="text-lg font-semibold text-white">Clinical Parameters</h2>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Demographics */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Patient Age
            </label>
            <div className="relative">
              <input
                type="number"
                min={1}
                max={120}
                value={age}
                onChange={(e) => setAge(Number(e.target.value))}
                className="input-field"
              />
              <span className="absolute right-3 top-2.5 text-xs text-slate-500">Yrs</span>
            </div>
          </div>
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Sex
            </label>
            <select
              value={sex}
              onChange={(e) => setSex(e.target.value)}
              className="input-field appearance-none"
            >
              <option>Male</option>
              <option>Female</option>
            </select>
          </div>
        </div>

        <hr className="border-slate-800" />

        {/* Genetic Markers */}
        <div>
          <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-3">
            Genetic Mutations
          </label>
          <div className="space-y-2">
            <MarkerToggle
              label="NPM1"
              description="Nucleophosmin 1 — favorable prognosis"
              checked={npm1}
              onChange={setNpm1}
            />
            <MarkerToggle
              label="FLT3-ITD"
              description="FMS-like tyrosine kinase 3 — adverse outcomes"
              checked={flt3}
              onChange={setFlt3}
            />
            <MarkerToggle
              label="Other (CEBPA, IDH1/2)"
              description="CEBPA, IDH1/2, TET2, or other identified mutations"
              checked={geneticOther}
              onChange={setGeneticOther}
            />
          </div>
        </div>

        {/* Actions */}
        <div className="pt-4 border-t border-slate-800">
          <div className="flex items-center justify-between">
            <button
              type="button"
              onClick={onBack}
              className="btn-ghost inline-flex items-center gap-1.5"
            >
              <ArrowLeft size={14} />
              Back
            </button>
            <button
              type="submit"
              className="btn-primary inline-flex items-center gap-2 shadow-glow-lg"
            >
              <span className="material-icons-outlined text-base animate-pulse">analytics</span>
              Run Diagnostic Analysis
            </button>
          </div>
          <p className="text-center text-xs text-slate-500 mt-3">
            Estimated processing time: ~45 seconds
          </p>
        </div>
      </form>
    </motion.div>
  );
}

function MarkerToggle({
  label,
  description,
  checked,
  onChange,
}: {
  label: string;
  description: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className={clsx(
        'w-full flex items-center justify-between p-3 rounded-lg transition-all duration-200 text-left',
        checked
          ? 'bg-primary/10 border border-primary/30'
          : 'bg-surface-light/30 border border-slate-800 hover:bg-surface-light/60'
      )}
    >
      <div>
        <div className={clsx('text-sm font-medium', checked ? 'text-white' : 'text-slate-300')}>
          {label}
        </div>
        <div className={clsx('text-xs mt-0.5', checked ? 'text-slate-400' : 'text-slate-500')}>
          {description}
        </div>
      </div>

      <div className="flex bg-slate-900 rounded p-1 ml-3 flex-shrink-0">
        <span
          className={clsx(
            'px-3 py-1 text-xs rounded font-medium transition-all',
            checked ? 'bg-slate-700 text-white shadow-sm' : 'text-slate-500'
          )}
        >
          Pos
        </span>
        <span
          className={clsx(
            'px-3 py-1 text-xs rounded font-medium transition-all',
            !checked ? 'bg-slate-700 text-white shadow-sm' : 'text-slate-500'
          )}
        >
          Neg
        </span>
      </div>
    </button>
  );
}

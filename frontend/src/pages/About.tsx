import { motion } from 'framer-motion';

const ease = [0.25, 0.1, 0.25, 1] as const;

function FadeIn({ children, delay = 0, className = '' }: { children: React.ReactNode; delay?: number; className?: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-40px' }}
      transition={{ duration: 0.65, delay, ease }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

/* ── Pipeline node component ──────────────────── */
function PipelineNode({ label, sub, icon, accent = '#137fec', highlight = false }: {
  label: string; sub: string; icon: string; accent?: string; highlight?: boolean;
}) {
  return (
    <div className={`flex items-center gap-3 px-4 py-3 rounded-xl border transition-all ${
      highlight ? 'border-opacity-40' : 'border-slate-700/50'
    }`}
      style={{
        background: highlight ? `${accent}08` : 'rgba(30,45,61,0.5)',
        borderColor: highlight ? accent : undefined,
      }}>
      <div className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0"
        style={{ background: `${accent}15` }}>
        <span className="material-icons-outlined text-lg" style={{ color: accent }}>{icon}</span>
      </div>
      <div className="min-w-0">
        <p className="text-sm font-semibold text-white leading-tight">{label}</p>
        <p className="text-xs text-slate-500 leading-tight mt-0.5">{sub}</p>
      </div>
    </div>
  );
}

function PipelineConnector() {
  return (
    <div className="flex justify-center py-2">
      <div className="w-px h-6 bg-gradient-to-b from-slate-700 to-slate-800" />
    </div>
  );
}

function PipelineFork() {
  return (
    <div className="flex justify-center py-2">
      <div className="relative w-48 h-8">
        {/* Left arm */}
        <div className="absolute left-1/4 top-0 w-px h-3" style={{ background: 'linear-gradient(to bottom, #2a3b4d, #334155)' }} />
        <div className="absolute left-1/4 top-3 right-1/4 h-px" style={{ background: 'linear-gradient(to right, #334155, #2a3b4d, #334155)' }} />
        {/* Right arm */}
        <div className="absolute right-1/4 top-0 w-px h-3" style={{ background: 'linear-gradient(to bottom, #2a3b4d, #334155)' }} />
        {/* Center stem */}
        <div className="absolute left-1/2 top-3 w-px h-5 -translate-x-px" style={{ background: 'linear-gradient(to bottom, #2a3b4d, #475569)' }} />
        {/* Dot indicator */}
        <div className="absolute left-1/2 bottom-0 w-1.5 h-1.5 rounded-full -translate-x-[3px]" style={{ background: '#475569' }} />
      </div>
    </div>
  );
}

export default function About() {
  return (
    <div className="min-h-screen">
      {/* ── Header ─────────────────────────────── */}
      <section className="section-container pt-20 pb-16">
        <FadeIn>
          <p className="text-primary text-sm font-medium tracking-tight mb-4">
            Under the hood
          </p>
          <h1 className="text-3xl sm:text-4xl font-bold text-white max-w-[560px] tracking-tight leading-tight">
            Built to earn clinical trust.
          </h1>
          <p className="text-base text-slate-400 max-w-[520px] mt-5 leading-relaxed">
            A hybrid multimodal architecture that auto-segments cells, extracts 20
            handcrafted morphological features, fuses them with deep CNN features
            — and explains every prediction with Grad-CAM.
          </p>
        </FadeIn>
      </section>

      {/* ── Architecture — HTML/CSS Pipeline ───── */}
      <section className="border-y border-slate-800/60">
        <div className="section-container py-16">
          <FadeIn className="mb-8">
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
              <span className="material-icons-outlined text-primary text-xl">account_tree</span>
              Model Architecture
            </h2>
          </FadeIn>

          <FadeIn delay={0.1}>
            <div className="max-w-xl mx-auto space-y-0">
              {/* Input */}
              <PipelineNode icon="image" label="Blood Slide Upload" sub="Single cell or whole smear field" accent="#137fec" />

              <PipelineConnector />

              {/* Auto-segmentation */}
              <PipelineNode icon="grid_view" label="Auto-Segmentation" sub="Otsu + contour detection → per-cell crops (224×224)" accent="#f59e0b" highlight />

              <PipelineConnector />

              {/* Dual streams */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <PipelineNode icon="hub" label="Visual Stream" sub="ResNet-50 → 2048-d" accent="#137fec" highlight />
                <PipelineNode icon="science" label="Morphology Stream" sub="20 features → MLP → 32-d" accent="#8b5cf6" highlight />
              </div>

              {/* Feature detail */}
              <div className="px-4 py-3 rounded-lg border border-slate-800/40 text-xs text-slate-500 leading-relaxed" style={{ background: 'rgba(139,92,246,0.03)' }}>
                <span className="text-slate-400 font-medium">20 morphological features:</span>{' '}
                <span className="text-slate-500">Geometry</span> → cell area, perimeter, circularity, eccentricity &nbsp;|&nbsp;
                <span className="text-slate-500">Nucleus</span> → nuclear area, N:C ratio, nuclear irregularity &nbsp;|&nbsp;
                <span className="text-slate-500">Colour</span> → RGB means (3), HSV means (3), stain intensity &amp; variance &nbsp;|&nbsp;
                <span className="text-slate-500">Texture</span> → GLCM contrast, homogeneity, energy, correlation &nbsp;|&nbsp;
                <span className="text-slate-500">Shape</span> → solidity
              </div>

              <PipelineFork />

              {/* Fusion */}
              <PipelineNode icon="merge" label="Late Fusion" sub="[2048 + 32] = 2080-dim concat" accent="#06b6d4" highlight />

              <PipelineConnector />

              {/* Classifier */}
              <div className="px-4 py-3.5 rounded-xl border border-emerald-500/30 text-center"
                style={{ background: 'rgba(34,197,94,0.04)' }}>
                <div className="flex items-center justify-center gap-2 mb-1">
                  <span className="material-icons-outlined text-emerald-400 text-lg">check_circle</span>
                  <p className="text-sm font-semibold text-white">Classifier Output</p>
                </div>
                <p className="text-xs text-slate-500">FC(2080→256) → BN → ReLU → Dropout → FC(256→1)</p>
                <span className="inline-block mt-1.5 px-3 py-0.5 rounded-full text-xs font-semibold text-emerald-400"
                  style={{ background: 'rgba(34,197,94,0.1)' }}>
                  P(AML) ∈ [0, 1] &nbsp; threshold = 0.786
                </span>
              </div>
            </div>
          </FadeIn>
        </div>
      </section>

      {/* ── Design Principles ──────────────────── */}
      <section className="py-20">
        <div className="section-container">
          <FadeIn className="mb-14">
            <h2 className="text-2xl font-bold text-white mb-3 tracking-tight">Six design principles.</h2>
            <p className="text-base text-slate-400 max-w-[480px]">
              Every architectural choice serves one goal: accurate, explainable,
              trustworthy diagnosis.
            </p>
          </FadeIn>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {[
              {
                num: '01',
                icon: 'grid_view',
                title: 'Auto Cell Segmentation',
                body: 'Upload a whole blood smear — Otsu thresholding and contour detection automatically isolate individual cells for per-cell analysis.',
                color: '#f59e0b',
              },
              {
                num: '02',
                icon: 'merge',
                title: 'Multimodal Fusion',
                body: 'Fuses 2048-dim ResNet-50 visual features with a 32-dim MLP encoding of 20 handcrafted morphological measurements via late concatenation.',
                color: '#137fec',
              },
              {
                num: '03',
                icon: 'visibility',
                title: 'Grad-CAM Explainability',
                body: 'Generates heatmaps from layer3 and layer4 of ResNet-50, highlighting which cell regions drove each prediction.',
                color: '#8b5cf6',
              },
              {
                num: '04',
                icon: 'call_split',
                title: 'Patient-Level Splitting',
                body: 'Train/val/test split by patient ID — all images from one patient stay in the same split, preventing data leakage.',
                color: '#06b6d4',
              },
              {
                num: '05',
                icon: 'balance',
                title: 'Class Imbalance Handling',
                body: 'Weighted BCE loss and weighted random sampling ensure balanced learning despite skewed class distribution.',
                color: '#10b981',
              },
              {
                num: '06',
                icon: 'tune',
                title: 'Optimized Threshold',
                body: 'Classification threshold (0.786) tuned on validation set by maximizing Youden\'s J-statistic for optimal sensitivity/specificity balance.',
                color: '#ef4444',
              },
            ].map((item, i) => (
              <FadeIn key={item.num} delay={i * 0.05}>
                <div className="p-5 rounded-xl border border-slate-800/60 h-full transition-all duration-200 hover:border-slate-700"
                  style={{ background: 'linear-gradient(180deg, #19232e 0%, #151e29 100%)' }}>
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0"
                      style={{ background: `${item.color}12`, border: `1px solid ${item.color}20` }}>
                      <span className="material-icons-outlined text-lg" style={{ color: item.color }}>{item.icon}</span>
                    </div>
                    <div>
                      <span className="text-[10px] font-bold text-slate-600 tracking-widest uppercase block leading-none mb-0.5">{item.num}</span>
                      <h3 className="text-sm font-semibold text-white leading-tight">{item.title}</h3>
                    </div>
                  </div>
                  <p className="text-sm text-slate-400 leading-relaxed">{item.body}</p>
                </div>
              </FadeIn>
            ))}
          </div>
        </div>
      </section>

      {/* ── Technology Stack ────────────────────── */}
      <section className="border-y border-slate-800/60">
        <div className="section-container py-16">
          <FadeIn className="mb-8">
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
              <span className="material-icons-outlined text-primary text-xl">layers</span>
              Technology Stack
            </h2>
          </FadeIn>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { name: 'PyTorch', role: 'ML Framework', icon: 'local_fire_department', color: '#f59e0b' },
              { name: 'ResNet-50', role: 'Backbone', icon: 'hub', color: '#137fec' },
              { name: 'FastAPI', role: 'Backend', icon: 'bolt', color: '#10b981' },
              { name: 'React', role: 'Frontend', icon: 'code', color: '#06b6d4' },
              { name: 'Tailwind CSS', role: 'Styling', icon: 'palette', color: '#8b5cf6' },
              { name: 'Grad-CAM', role: 'Explainability', icon: 'visibility', color: '#ef4444' },
              { name: 'Gradio', role: 'Demo', icon: 'web', color: '#f59e0b' },
              { name: 'TypeScript', role: 'Language', icon: 'data_object', color: '#3b82f6' },
            ].map((tech, i) => (
              <FadeIn key={tech.name} delay={i * 0.03}>
                <div className="py-4 px-4 rounded-xl border border-slate-800/60 text-center hover:border-slate-700 transition-colors"
                  style={{ background: 'linear-gradient(180deg, #19232e 0%, #151e29 100%)' }}>
                  <span className="material-icons-outlined text-xl mb-1.5 block" style={{ color: tech.color }}>{tech.icon}</span>
                  <div className="text-sm font-semibold text-white">{tech.name}</div>
                  <div className="text-xs text-slate-500 mt-0.5">{tech.role}</div>
                </div>
              </FadeIn>
            ))}
          </div>
        </div>
      </section>

      {/* ── Dataset ─────────────────────────────── */}
      <section className="py-16">
        <div className="section-container">
          <FadeIn>
            <h2 className="text-lg font-semibold text-white flex items-center gap-2 mb-4">
              <span className="material-icons-outlined text-primary text-xl">science</span>
              Dataset
            </h2>
            <p className="text-sm text-slate-300 leading-relaxed max-w-[580px] mb-4">
              Trained on the <strong className="font-semibold text-white">Munich AML-Cytomorphology</strong> dataset
              from LMU Munich, containing 18,000+ single-cell microscopy images from 200+ patients
              with expert annotations. Binary classification: blast cells vs. normal haematopoietic cells.
            </p>
            <a
              href="https://www.cancerimagingarchive.net/"
              target="_blank"
              rel="noopener noreferrer"
              className="group inline-flex items-center gap-1.5 text-sm text-primary font-medium transition-colors"
            >
              <span className="relative">
                View on TCIA
                <span className="absolute bottom-0 left-0 w-0 h-px bg-primary transition-all duration-300 ease-out group-hover:w-full" />
              </span>
              <span className="material-icons-outlined transition-transform duration-200 group-hover:translate-x-0.5 group-hover:-translate-y-0.5" style={{ fontSize: '14px' }}>arrow_outward</span>
            </a>
          </FadeIn>
        </div>
      </section>

      {/* ── Disclaimer ──────────────────────────── */}
      <section className="border-t border-slate-800/60">
        <div className="section-container py-14">
          <FadeIn>
            <div className="max-w-[580px] p-5 rounded-xl border border-amber-500/10"
              style={{ background: 'rgba(245,158,11,0.03)' }}>
              <div className="flex items-center gap-2 mb-2">
                <span className="material-icons-outlined text-amber-400" style={{ fontSize: '18px' }}>warning_amber</span>
                <p className="text-sm font-semibold text-white">Medical Disclaimer</p>
              </div>
              <p className="text-sm text-slate-400 leading-relaxed">
                HemaVision is a research and educational tool. It is <strong className="font-medium text-white">not</strong> intended
                for clinical diagnosis or to replace professional medical judgment. Always consult
                qualified hematologists and follow institutional protocols for patient care decisions.
              </p>
            </div>
          </FadeIn>
        </div>
      </section>
    </div>
  );
}

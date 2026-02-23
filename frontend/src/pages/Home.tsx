import { motion, useScroll, useTransform } from 'framer-motion';
import { Link } from 'react-router-dom';
import { useRef, useEffect, useState } from 'react';
import { getMetrics, type PlatformMetrics } from '../lib/api';

const ease = [0.25, 0.1, 0.25, 1] as const;

function FadeIn({ children, delay = 0, className = '' }: { children: React.ReactNode; delay?: number; className?: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-60px' }}
      transition={{ duration: 0.7, delay, ease }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

/* ── Default metrics (used while loading / if API fails) ── */
const DEFAULT_METRICS: PlatformMetrics = {
  accuracy: 96.8,
  auc_roc: 0.976,
  precision: 90.0,
  recall: 90.0,
  f1_score: 90.0,
  inference_ms: 50,
  dataset_size: 18577,
  dataset_patients: 200,
  dataset_source: 'Munich AML-Cytomorphology (TCIA)',
  model_version: 'DualStream v2.4',
  last_trained: null,
};

/* ── OpenAI-style Abstract Art Background ─── */
function AbstractArt() {
  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      {/* Flowing gradient mesh */}
      <svg className="absolute inset-0 w-full h-full" viewBox="0 0 1200 600" fill="none" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid slice">
        <defs>
          <linearGradient id="artGrad1" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#137fec" stopOpacity="0.08" />
            <stop offset="50%" stopColor="#7c3aed" stopOpacity="0.05" />
            <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.03" />
          </linearGradient>
          <linearGradient id="artGrad2" x1="100%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#a78bfa" stopOpacity="0.06" />
            <stop offset="100%" stopColor="#137fec" stopOpacity="0.02" />
          </linearGradient>
          <filter id="artBlur">
            <feGaussianBlur stdDeviation="40" />
          </filter>
        </defs>
        {/* Organic flowing shapes */}
        <path d="M0 300 Q200 100 400 250 T800 200 T1200 300 L1200 600 L0 600Z" fill="url(#artGrad1)" filter="url(#artBlur)">
          <animate attributeName="d" dur="20s" repeatCount="indefinite"
            values="M0 300 Q200 100 400 250 T800 200 T1200 300 L1200 600 L0 600Z;
                    M0 250 Q200 200 400 150 T800 300 T1200 250 L1200 600 L0 600Z;
                    M0 300 Q200 100 400 250 T800 200 T1200 300 L1200 600 L0 600Z" />
        </path>
        <path d="M0 400 Q300 200 600 350 T1200 400 L1200 600 L0 600Z" fill="url(#artGrad2)" filter="url(#artBlur)">
          <animate attributeName="d" dur="25s" repeatCount="indefinite"
            values="M0 400 Q300 200 600 350 T1200 400 L1200 600 L0 600Z;
                    M0 350 Q300 300 600 250 T1200 350 L1200 600 L0 600Z;
                    M0 400 Q300 200 600 350 T1200 400 L1200 600 L0 600Z" />
        </path>
        {/* Floating orbs */}
        <circle cx="200" cy="150" r="80" fill="#137fec" fillOpacity="0.04" filter="url(#artBlur)">
          <animate attributeName="cy" values="150;200;150" dur="12s" repeatCount="indefinite" />
        </circle>
        <circle cx="900" cy="250" r="100" fill="#7c3aed" fillOpacity="0.03" filter="url(#artBlur)">
          <animate attributeName="cx" values="900;950;900" dur="15s" repeatCount="indefinite" />
        </circle>
        <circle cx="600" cy="100" r="60" fill="#06b6d4" fillOpacity="0.04" filter="url(#artBlur)">
          <animate attributeName="cy" values="100;140;100" dur="10s" repeatCount="indefinite" />
        </circle>
      </svg>
      {/* Grid lines — subtle structure */}
      <div className="absolute inset-0 opacity-[0.02]"
        style={{ backgroundImage: 'linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)', backgroundSize: '60px 60px' }} />
      {/* Noise texture */}
      <div className="absolute inset-0 opacity-[0.03]"
        style={{ backgroundImage: 'url("data:image/svg+xml,%3Csvg viewBox=\'0 0 256 256\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cfilter id=\'n\'%3E%3CfeTurbulence baseFrequency=\'0.9\' stitchTiles=\'stitch\'/%3E%3C/filter%3E%3Crect width=\'100%25\' height=\'100%25\' filter=\'url(%23n)\' opacity=\'0.5\'/%3E%3C/svg%3E")', backgroundSize: '128px 128px' }} />
    </div>
  );
}

/* ── Blood Cell Scanner Visualization ─────── */
function CellVisualization() {
  const scanPoints = [
    { x: 85, y: 80 },
    { x: 200, y: 65 },
    { x: 150, y: 150 },
    { x: 220, y: 190 },
    { x: 70, y: 200 },
    { x: 180, y: 120 },
  ];
  const pathD = scanPoints.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ') + ` L${scanPoints[0].x},${scanPoints[0].y}`;

  return (
    <div className="relative w-full h-full">
      <div className="absolute inset-0 rounded-full bg-gradient-to-br from-primary/10 via-violet-500/5 to-transparent blur-2xl animate-breathe" />
      <svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg" className="relative w-full h-full">
        <defs>
          <radialGradient id="rbcFill" cx="40%" cy="35%" r="55%">
            <stop offset="0%" stopColor="#ef4444" stopOpacity="0.4" />
            <stop offset="60%" stopColor="#dc2626" stopOpacity="0.2" />
            <stop offset="100%" stopColor="#991b1b" stopOpacity="0.06" />
          </radialGradient>
          <radialGradient id="rbcHighlight" cx="30%" cy="25%" r="30%">
            <stop offset="0%" stopColor="#fca5a5" stopOpacity="0.3" />
            <stop offset="100%" stopColor="#ef4444" stopOpacity="0" />
          </radialGradient>
          <radialGradient id="wbcFill" cx="40%" cy="35%" r="55%">
            <stop offset="0%" stopColor="#c4b5fd" stopOpacity="0.55" />
            <stop offset="60%" stopColor="#a78bfa" stopOpacity="0.3" />
            <stop offset="100%" stopColor="#6d28d9" stopOpacity="0.08" />
          </radialGradient>
          <radialGradient id="wbcHighlight" cx="30%" cy="25%" r="25%">
            <stop offset="0%" stopColor="#e9d5ff" stopOpacity="0.3" />
            <stop offset="100%" stopColor="#a78bfa" stopOpacity="0" />
          </radialGradient>
          <radialGradient id="plateletFill" cx="50%" cy="40%" r="50%">
            <stop offset="0%" stopColor="#fde68a" stopOpacity="0.55" />
            <stop offset="100%" stopColor="#d97706" stopOpacity="0.08" />
          </radialGradient>
          <filter id="cellGlow">
            <feGaussianBlur stdDeviation="3" result="b" />
            <feComposite in="SourceGraphic" in2="b" operator="over" />
          </filter>
          <filter id="cellShadow">
            <feDropShadow dx="0" dy="1" stdDeviation="3" floodColor="#000" floodOpacity="0.3" />
          </filter>
          <filter id="pointerGlow">
            <feGaussianBlur stdDeviation="5" result="b" />
            <feComposite in="SourceGraphic" in2="b" operator="over" />
          </filter>
          <linearGradient id="scanLine" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#137fec" stopOpacity="0" />
            <stop offset="50%" stopColor="#137fec" stopOpacity="0.12" />
            <stop offset="100%" stopColor="#137fec" stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* Grid pattern — subtle diagnostic field */}
        <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
          <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#1e2d3d" strokeWidth="0.25" />
        </pattern>
        <circle cx="150" cy="150" r="138" fill="url(#grid)" />

        {/* Scanning line — sweeps vertically */}
        <rect x="12" y="0" width="276" height="30" fill="url(#scanLine)" rx="2">
          <animate attributeName="y" values="12;268;12" dur="4s" repeatCount="indefinite" />
        </rect>

        {/* Outer ring — microscope field of view */}
        <circle cx="150" cy="150" r="138" fill="none" stroke="#1e2d3d" strokeWidth="1" />
        {/* Inner faint ring */}
        <circle cx="150" cy="150" r="130" fill="none" stroke="#1e2d3d" strokeWidth="0.4" strokeOpacity="0.5" />
        <circle cx="150" cy="150" r="138" fill="none" stroke="#137fec" strokeWidth="1.5" strokeOpacity="0.15"
          strokeDasharray="8 12" strokeLinecap="round">
          <animateTransform attributeName="transform" type="rotate" from="0 150 150" to="360 150 150" dur="30s" repeatCount="indefinite" />
        </circle>
        {/* Counter-rotating tick marks */}
        <circle cx="150" cy="150" r="134" fill="none" stroke="#137fec" strokeWidth="0.6" strokeOpacity="0.08"
          strokeDasharray="2 18" strokeLinecap="round">
          <animateTransform attributeName="transform" type="rotate" from="360 150 150" to="0 150 150" dur="45s" repeatCount="indefinite" />
        </circle>

        {/* ── Red Blood Cells (biconcave discs with highlight) ── */}
        {[
          { cx: 200, cy: 65, rx: 22, ry: 18 },
          { cx: 70, cy: 200, rx: 20, ry: 16 },
          { cx: 180, cy: 120, rx: 18, ry: 14 },
          { cx: 120, cy: 240, rx: 19, ry: 15 },
          { cx: 240, cy: 160, rx: 17, ry: 13 },
          { cx: 55, cy: 120, rx: 16, ry: 13 },
          { cx: 230, cy: 240, rx: 18, ry: 14 },
          { cx: 95, cy: 60, rx: 15, ry: 12 },
        ].map((rbc, i) => (
          <g key={`rbc-${i}`} filter="url(#cellShadow)">
            {/* Cell body */}
            <ellipse cx={rbc.cx} cy={rbc.cy} rx={rbc.rx} ry={rbc.ry}
              fill="url(#rbcFill)" stroke="#ef4444" strokeWidth="0.6" strokeOpacity="0.3" />
            {/* Membrane shimmer highlight */}
            <ellipse cx={rbc.cx - rbc.rx * 0.15} cy={rbc.cy - rbc.ry * 0.15} rx={rbc.rx * 0.85} ry={rbc.ry * 0.8}
              fill="url(#rbcHighlight)" />
            {/* Inner dimple — biconcave center */}
            <ellipse cx={rbc.cx} cy={rbc.cy} rx={rbc.rx * 0.42} ry={rbc.ry * 0.38}
              fill="none" stroke="#ef4444" strokeWidth="0.5" strokeOpacity="0.18" />
            <ellipse cx={rbc.cx} cy={rbc.cy} rx={rbc.rx * 0.2} ry={rbc.ry * 0.18}
              fill="#991b1b" fillOpacity="0.08" />
          </g>
        ))}

        {/* ── White Blood Cells (larger, lobed nucleus with highlight) ── */}
        {[
          { cx: 85, cy: 80, r: 20 },
          { cx: 150, cy: 150, r: 24 },
        ].map((wbc, i) => (
          <g key={`wbc-${i}`} filter="url(#cellGlow)">
            {/* Cell body */}
            <circle cx={wbc.cx} cy={wbc.cy} r={wbc.r}
              fill="url(#wbcFill)" stroke="#a78bfa" strokeWidth="0.8" strokeOpacity="0.4" />
            {/* Membrane shimmer */}
            <circle cx={wbc.cx - 2} cy={wbc.cy - 2} r={wbc.r * 0.85}
              fill="url(#wbcHighlight)" />
            {/* Multi-lobed nucleus with connecting bridges */}
            <circle cx={wbc.cx - 5} cy={wbc.cy - 3} r={wbc.r * 0.32}
              fill="#7c3aed" fillOpacity="0.35" stroke="#a78bfa" strokeWidth="0.5" strokeOpacity="0.35" />
            <circle cx={wbc.cx + 5} cy={wbc.cy + 2} r={wbc.r * 0.28}
              fill="#7c3aed" fillOpacity="0.3" stroke="#a78bfa" strokeWidth="0.5" strokeOpacity="0.3" />
            <circle cx={wbc.cx + 1} cy={wbc.cy - 7} r={wbc.r * 0.22}
              fill="#7c3aed" fillOpacity="0.25" stroke="#a78bfa" strokeWidth="0.3" strokeOpacity="0.2" />
            {/* Nucleus bridge */}
            <line x1={wbc.cx - 3} y1={wbc.cy - 2} x2={wbc.cx + 3} y2={wbc.cy + 1}
              stroke="#7c3aed" strokeWidth="1.5" strokeOpacity="0.15" strokeLinecap="round" />
            {/* Granules in cytoplasm */}
            {[...Array(5)].map((_, j) => {
              const angle = (j / 5) * Math.PI * 2;
              const dist = wbc.r * 0.6;
              return (
                <circle key={j} cx={wbc.cx + Math.cos(angle) * dist} cy={wbc.cy + Math.sin(angle) * dist}
                  r="1.2" fill="#a78bfa" fillOpacity="0.15" />
              );
            })}
          </g>
        ))}

        {/* ── Platelets (irregular clusters with granules) ── */}
        {[
          { cx: 220, cy: 190 }, { cx: 225, cy: 200 }, { cx: 215, cy: 195 },
          { cx: 130, cy: 110 }, { cx: 135, cy: 105 },
          { cx: 170, cy: 220 }, { cx: 165, cy: 225 },
        ].map((p, i) => (
          <g key={`pl-${i}`}>
            <circle cx={p.cx} cy={p.cy} r={3.5 + (i % 2)}
              fill="url(#plateletFill)" stroke="#fbbf24" strokeWidth="0.4" strokeOpacity="0.35" />
            {/* Granule dot */}
            <circle cx={p.cx + 0.5} cy={p.cy - 0.5} r="1" fill="#fde68a" fillOpacity="0.4" />
          </g>
        ))}

        {/* ── Scanner pointer path (dashed trail) ── */}
        <path d={pathD} fill="none" stroke="#137fec" strokeWidth="0.5" strokeOpacity="0.1" strokeDasharray="4 8" />

        {/* ── Scanning pointer that moves along path ── */}
        <g filter="url(#pointerGlow)">
          {/* Pointer glow ring */}
          <circle r="16" fill="none" stroke="#137fec" strokeWidth="1" strokeOpacity="0.3" strokeDasharray="6 4">
            <animateMotion dur="8s" repeatCount="indefinite" path={pathD} />
            <animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="3s" repeatCount="indefinite" additive="sum" />
          </circle>
          {/* Pointer crosshair - horizontal */}
          <line x1="-10" y1="0" x2="-5" y2="0" stroke="#60a5fa" strokeWidth="1" strokeOpacity="0.6">
            <animateMotion dur="8s" repeatCount="indefinite" path={pathD} />
          </line>
          <line x1="5" y1="0" x2="10" y2="0" stroke="#60a5fa" strokeWidth="1" strokeOpacity="0.6">
            <animateMotion dur="8s" repeatCount="indefinite" path={pathD} />
          </line>
          {/* Pointer crosshair - vertical */}
          <line x1="0" y1="-10" x2="0" y2="-5" stroke="#60a5fa" strokeWidth="1" strokeOpacity="0.6">
            <animateMotion dur="8s" repeatCount="indefinite" path={pathD} />
          </line>
          <line x1="0" y1="5" x2="0" y2="10" stroke="#60a5fa" strokeWidth="1" strokeOpacity="0.6">
            <animateMotion dur="8s" repeatCount="indefinite" path={pathD} />
          </line>
          {/* Center dot */}
          <circle r="2.5" fill="#60a5fa" fillOpacity="0.9">
            <animateMotion dur="8s" repeatCount="indefinite" path={pathD} />
          </circle>
          {/* Pulse ring */}
          <circle r="8" fill="none" stroke="#137fec" strokeWidth="1.5" strokeOpacity="0.4">
            <animateMotion dur="8s" repeatCount="indefinite" path={pathD} />
            <animate attributeName="r" values="8;18;8" dur="1.5s" repeatCount="indefinite" />
            <animate attributeName="stroke-opacity" values="0.4;0;0.4" dur="1.5s" repeatCount="indefinite" />
          </circle>
        </g>

        {/* ── HUD overlays ── */}
        {/* Corner brackets — refined with glow */}
        {[
          'M22 38 L22 22 L38 22',
          'M262 22 L278 22 L278 38',
          'M22 262 L22 278 L38 278',
          'M262 278 L278 278 L278 262',
        ].map((d, i) => (
          <g key={`bracket-${i}`}>
            <path d={d} fill="none" stroke="#137fec" strokeWidth="1.5" strokeOpacity="0.12" strokeLinecap="round" />
            <path d={d} fill="none" stroke="#137fec" strokeWidth="0.8" strokeOpacity="0.4" strokeLinecap="round" />
          </g>
        ))}

        {/* Crosshairs — faint center lines */}
        <line x1="150" y1="18" x2="150" y2="30" stroke="#137fec" strokeWidth="0.4" strokeOpacity="0.15" />
        <line x1="150" y1="270" x2="150" y2="282" stroke="#137fec" strokeWidth="0.4" strokeOpacity="0.15" />
        <line x1="18" y1="150" x2="30" y2="150" stroke="#137fec" strokeWidth="0.4" strokeOpacity="0.15" />
        <line x1="270" y1="150" x2="282" y2="150" stroke="#137fec" strokeWidth="0.4" strokeOpacity="0.15" />

        {/* Status label — pill */}
        <rect x="95" y="270" width="110" height="18" rx="9" fill="#137fec" fillOpacity="0.08" stroke="#137fec" strokeWidth="0.5" strokeOpacity="0.2" />
        <text x="150" y="282" textAnchor="middle" fill="#60a5fa" fontSize="8" fontWeight="600" fontFamily="JetBrains Mono, monospace" letterSpacing="0.05em">
          SCANNING...
        </text>

        {/* Top-left data readout */}
        <text x="30" y="46" fill="#475569" fontSize="5.5" fontFamily="JetBrains Mono, monospace" letterSpacing="0.03em">RBC: 8  WBC: 2  PLT: 7</text>

        {/* Detection badge on center WBC */}
        <rect x="157" y="130" width="42" height="16" rx="8" fill="#137fec" fillOpacity="0.12" stroke="#137fec" strokeWidth="0.6" strokeOpacity="0.3">
          <animate attributeName="fill-opacity" values="0.12;0.22;0.12" dur="2s" repeatCount="indefinite" />
        </rect>
        <text x="178" y="141" textAnchor="middle" fill="#60a5fa" fontSize="7.5" fontWeight="700" fontFamily="JetBrains Mono, monospace">AML?</text>
        {/* Detection line from badge to WBC */}
        <line x1="157" y1="138" x2="150" y2="150" stroke="#137fec" strokeWidth="0.4" strokeOpacity="0.2" strokeDasharray="2 2" />
      </svg>
    </div>
  );
}

export default function Home() {
  const heroRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: heroRef,
    offset: ['start start', 'end start'],
  });
  const heroOpacity = useTransform(scrollYProgress, [0, 1], [1, 0]);
  const heroScale = useTransform(scrollYProgress, [0, 1], [1, 0.97]);

  // Fetch dynamic metrics from API
  const [metrics, setMetrics] = useState<PlatformMetrics>(DEFAULT_METRICS);
  useEffect(() => {
    getMetrics().then(setMetrics).catch(() => { /* use defaults */ });
  }, []);

  return (
    <div className="space-y-5 sm:space-y-8 py-6 sm:py-8">
      <div className="section-container space-y-5 sm:space-y-8">
        {/* ═══════════════════════════════════
            HERO
            ═══════════════════════════════════ */}
        <section
          ref={heroRef}
          className="relative overflow-hidden rounded-2xl border border-slate-800/60"
          style={{ background: 'linear-gradient(135deg, #19232e 0%, #101922 40%, #0d1a2a 100%)' }}
        >
          {/* Abstract art background */}
          <AbstractArt />

          <motion.div
            style={{ opacity: heroOpacity, scale: heroScale }}
            className="relative z-10 p-6 sm:p-8 md:p-12 lg:p-14 flex flex-col md:flex-row items-start md:items-center justify-between gap-8 md:gap-10"
          >
            <div className="max-w-2xl space-y-6">
              {/* Status badge + CTA together */}
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.1, ease }}
                className="flex flex-wrap items-center gap-3"
              >
                <div className="inline-flex items-center gap-2.5 px-4 py-1.5 rounded-full border border-primary/30 text-xs font-semibold uppercase tracking-wider"
                  style={{ background: 'linear-gradient(135deg, rgba(19,127,236,0.15) 0%, rgba(124,58,237,0.1) 100%)', color: '#60a5fa' }}>
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75" />
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-primary" />
                  </span>
                  AI Diagnostics Ready
                </div>
                <Link
                  to="/analyze"
                  className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-full text-xs font-semibold transition-all duration-200 hover:scale-105"
                  style={{ background: 'linear-gradient(135deg, #137fec 0%, #0e6adb 100%)', color: '#fff' }}
                >
                  <span className="material-icons-outlined" style={{ fontSize: '14px' }}>play_arrow</span>
                  Start Analysis
                </Link>
              </motion.div>

              {/* Headline */}
              <motion.h1
                initial={{ opacity: 0, y: 28 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.2, ease }}
                className="text-hero text-white leading-tight"
              >
                Precision AML{' '}
                <span className="text-transparent bg-clip-text"
                  style={{ backgroundImage: 'linear-gradient(135deg, #137fec 0%, #06b6d4 50%, #a78bfa 100%)' }}>
                  Diagnostics.
                </span>
              </motion.h1>

              {/* Subtitle */}
              <motion.p
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.35, ease }}
                className="text-base sm:text-lg text-slate-300/90 max-w-xl leading-relaxed"
              >
                Upload a blood smear, and the system auto-segments cells, extracts
                morphological features, and delivers explainable blast detection in under a second.
              </motion.p>

              {/* Live stats row — dynamic from API */}
              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.5, ease }}
                className="pt-2 flex flex-wrap gap-4 sm:gap-6"
              >
                {[
                  { value: `${metrics.accuracy}%`, label: 'Accuracy', icon: 'check_circle', color: '#4ade80' },
                  { value: `<${metrics.inference_ms}ms`, label: 'Latency', icon: 'bolt', color: '#f59e0b' },
                  { value: `${metrics.dataset_size >= 1000 ? `${(metrics.dataset_size / 1000).toFixed(0)}K+` : metrics.dataset_size}`, label: 'Training Cells', icon: 'science', color: '#60a5fa' },
                ].map((stat) => (
                  <div key={stat.label} className="flex items-center gap-2">
                    <span className="material-icons-outlined" style={{ fontSize: '16px', color: stat.color }}>{stat.icon}</span>
                    <div>
                      <span className="text-white font-bold text-sm tabular-nums">{stat.value}</span>
                      <span className="text-slate-500 text-xs ml-1.5">{stat.label}</span>
                    </div>
                  </div>
                ))}
              </motion.div>
            </div>

            {/* Hero visual — abstract cell SVG */}
            <motion.div
              initial={{ opacity: 0, scale: 0.85 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 1, delay: 0.4, ease }}
              className="hidden md:block w-52 h-52 lg:w-72 lg:h-72 flex-shrink-0"
            >
              <CellVisualization />
            </motion.div>
          </motion.div>
        </section>

        {/* ═══════════════════════════════════
            STATS & METRICS
            ═══════════════════════════════════ */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-5">
          {/* Quick Stat: Dataset */}
          <FadeIn delay={0}>
            <div className="h-full p-6 rounded-xl border border-slate-800/60 shadow-card flex flex-col justify-between"
              style={{ background: 'linear-gradient(180deg, #19232e 0%, #151e29 100%)' }}>
              <div className="flex justify-between items-start">
                <div>
                  <p className="text-sm font-medium text-slate-400">Dataset Size</p>
                  <h3 className="text-3xl font-bold text-white mt-1 tabular-nums">
                    {metrics.dataset_size >= 1000 ? `${(metrics.dataset_size / 1000).toFixed(0)}K+` : metrics.dataset_size}
                  </h3>
                </div>
                <div className="w-10 h-10 rounded-lg flex items-center justify-center" style={{ background: 'rgba(6,182,212,0.1)' }}>
                  <span className="material-icons-outlined text-xl text-cyan-400">science</span>
                </div>
              </div>
              <div>
                <p className="text-xs text-slate-500 mt-3">Single-cell images from {metrics.dataset_patients}+ patients</p>
                <p className="text-xs text-slate-600 mt-1">{metrics.dataset_source}</p>
              </div>
            </div>
          </FadeIn>

          {/* Stat: Accuracy */}
          <FadeIn delay={0.08}>
            <div className="h-full p-6 rounded-xl border border-slate-800/60 shadow-card flex flex-col justify-between"
              style={{ background: 'linear-gradient(180deg, #19232e 0%, #151e29 100%)' }}>
              <div className="flex justify-between items-start">
                <div>
                  <p className="text-sm font-medium text-slate-400">Model Accuracy</p>
                  <h3 className="text-3xl font-bold text-white mt-1 tabular-nums">{metrics.accuracy}%</h3>
                </div>
                <span className="px-2.5 py-1 rounded-md text-xs font-semibold flex items-center gap-1"
                  style={{ background: 'rgba(34,197,94,0.12)', color: '#4ade80', border: '1px solid rgba(34,197,94,0.2)' }}>
                  <span className="material-icons-outlined" style={{ fontSize: '11px' }}>check_circle</span>
                  Validated
                </span>
              </div>
              <div>
                <div className="w-full h-2 rounded-full mt-4 overflow-hidden" style={{ background: '#1e2d3d' }}>
                  <motion.div
                    initial={{ width: 0 }}
                    whileInView={{ width: `${metrics.accuracy}%` }}
                    viewport={{ once: true }}
                    transition={{ duration: 1.5, ease: 'easeOut', delay: 0.3 }}
                    className="h-2 rounded-full"
                    style={{ background: 'linear-gradient(90deg, #137fec 0%, #06b6d4 100%)' }}
                  />
                </div>
                <p className="text-xs text-slate-500 mt-2 tabular-nums">AUC-ROC: {metrics.auc_roc}</p>
              </div>
            </div>
          </FadeIn>

          {/* Stat: Inference Speed */}
          <FadeIn delay={0.16}>
            <div className="h-full p-6 rounded-xl border border-slate-800/60 shadow-card flex flex-col justify-between"
              style={{ background: 'linear-gradient(180deg, #19232e 0%, #151e29 100%)' }}>
              <div className="flex justify-between items-start">
                <div>
                  <p className="text-sm font-medium text-slate-400">Inference Speed</p>
                  <h3 className="text-3xl font-bold text-white mt-1 tabular-nums">&lt;{metrics.inference_ms}ms</h3>
                </div>
                <div className="w-10 h-10 rounded-lg flex items-center justify-center" style={{ background: 'rgba(245,158,11,0.1)' }}>
                  <span className="material-icons-outlined text-xl text-amber-400">bolt</span>
                </div>
              </div>
              <div>
                <p className="text-xs text-slate-500 mt-3">Per-cell classification on GPU</p>
                <p className="text-xs text-slate-600 mt-1">ResNet-50 + MLP fusion</p>
              </div>
            </div>
          </FadeIn>
        </div>

        {/* ── Performance Breakdown ─────────── */}
        <FadeIn delay={0.2}>
          <div className="p-6 rounded-xl border border-slate-800/60 shadow-card"
            style={{ background: 'linear-gradient(180deg, #19232e 0%, #151e29 100%)' }}>
            <h3 className="text-sm font-medium text-slate-400 mb-6">Performance Breakdown</h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 sm:gap-6">
              {[
                { label: 'Precision', value: metrics.precision, color: '#10b981', gradient: 'linear-gradient(135deg, #10b981, #059669)' },
                { label: 'Recall', value: metrics.recall, color: '#137fec', gradient: 'linear-gradient(135deg, #137fec, #0e6adb)' },
                { label: 'F1 Score', value: metrics.f1_score, color: '#8b5cf6', gradient: 'linear-gradient(135deg, #8b5cf6, #7c3aed)' },
                { label: 'AUC-ROC', value: metrics.auc_roc * 100, color: '#f59e0b', gradient: 'linear-gradient(135deg, #f59e0b, #d97706)' },
              ].map((metric, i) => (
                <div key={metric.label} className="text-center space-y-3">
                  {/* Circular progress ring */}
                  <div className="relative w-20 h-20 mx-auto">
                    <svg className="w-20 h-20 -rotate-90" viewBox="0 0 80 80">
                      <circle cx="40" cy="40" r="34" fill="none" stroke="#1e2d3d" strokeWidth="5" />
                      <motion.circle
                        cx="40" cy="40" r="34" fill="none"
                        stroke={metric.color}
                        strokeWidth="5"
                        strokeLinecap="round"
                        strokeDasharray={`${2 * Math.PI * 34}`}
                        initial={{ strokeDashoffset: 2 * Math.PI * 34 }}
                        whileInView={{ strokeDashoffset: 2 * Math.PI * 34 * (1 - metric.value / 100) }}
                        viewport={{ once: true }}
                        transition={{ duration: 1.2, ease: 'easeOut', delay: 0.3 + i * 0.12 }}
                      />
                    </svg>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <span className="text-lg font-bold text-white tabular-nums">
                        {metric.label === 'AUC-ROC' ? metrics.auc_roc.toFixed(2) : `${metric.value.toFixed(0)}%`}
                      </span>
                    </div>
                  </div>
                  <p className="text-xs font-medium text-slate-400">{metric.label}</p>
                </div>
              ))}
            </div>
          </div>
        </FadeIn>

        {/* ═══════════════════════════════════
            FEATURES — with colored accents + subtle art
            ═══════════════════════════════════ */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 relative">
          {/* Very subtle abstract orb behind features */}
          <div className="absolute -top-20 left-1/2 -translate-x-1/2 w-[500px] h-[300px] rounded-full opacity-[0.04] pointer-events-none"
            style={{ background: 'radial-gradient(ellipse, #137fec 0%, transparent 70%)' }} />
          {[
            {
              icon: 'hub',
              title: 'Dual-Stream Fusion',
              body: 'ResNet-50 encodes visual features. A parallel MLP encodes 20 handcrafted morphological measurements. Late fusion preserves each signal.',
              color: '#137fec',
            },
            {
              icon: 'visibility',
              title: 'Grad-CAM Explainability',
              body: 'Every prediction includes a heatmap showing exactly which cell regions drove the model\'s decision. A glass box, not a black one.',
              color: '#8b5cf6',
            },
            {
              icon: 'shield',
              title: 'Patient-Level Integrity',
              body: 'Train/test splits by patient ID, not image — preventing data leakage and ensuring predictions generalize to unseen patients.',
              color: '#10b981',
            },
          ].map((feature, i) => (
            <FadeIn key={feature.title} delay={i * 0.1}>
              <div className="h-full p-6 rounded-xl border border-slate-800/60 transition-all duration-300 hover:translate-y-[-2px] hover:shadow-card-hover group"
                style={{ background: 'linear-gradient(180deg, #19232e 0%, #151e29 100%)' }}>
                <div className="w-10 h-10 rounded-lg flex items-center justify-center mb-4 transition-transform duration-300 group-hover:scale-110"
                  style={{ background: `${feature.color}15`, border: `1px solid ${feature.color}25` }}>
                  <span className="material-icons-outlined text-xl" style={{ color: feature.color }}>{feature.icon}</span>
                </div>
                <h3 className="text-base font-semibold text-white mb-2">{feature.title}</h3>
                <p className="text-sm text-slate-400 leading-relaxed">{feature.body}</p>
              </div>
            </FadeIn>
          ))}
        </div>

        {/* ═══════════════════════════════════
            HOW IT WORKS
            ═══════════════════════════════════ */}
        <section className="rounded-xl p-6 sm:p-8 md:p-10 border border-slate-800/60"
          style={{ background: 'linear-gradient(180deg, #19232e 0%, #151e29 100%)' }}>
          <FadeIn className="mb-8">
            <h2 className="text-display text-white">
              Three steps. Under a second.
            </h2>
          </FadeIn>

          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                step: '01',
                icon: 'photo_camera',
                title: 'Upload a blood slide',
                body: 'Single-cell crop or whole blood smear field. The system auto-detects and segments individual cells.',
                color: '#137fec',
              },
              {
                step: '02',
                icon: 'memory',
                title: 'AI extracts & analyzes',
                body: '20 morphological features extracted per cell — area, perimeter, texture, color — fed alongside ResNet-50 visual features.',
                color: '#8b5cf6',
              },
              {
                step: '03',
                icon: 'insights',
                title: 'Receive explained diagnosis',
                body: 'Per-cell blast/normal classification with confidence score, risk level, and Grad-CAM heatmap overlay.',
                color: '#10b981',
              },
            ].map((s, i) => (
              <FadeIn key={s.step} delay={i * 0.1}>
                <div className="p-5 rounded-xl space-y-3"
                  style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.04)' }}>
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg flex items-center justify-center"
                      style={{ background: `${s.color}12`, border: `1px solid ${s.color}20` }}>
                      <span className="material-icons-outlined text-xl" style={{ color: s.color }}>{s.icon}</span>
                    </div>
                    <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: s.color }}>
                      Step {s.step}
                    </span>
                  </div>
                  <h3 className="text-sm font-semibold text-white">{s.title}</h3>
                  <p className="text-sm text-slate-400 leading-relaxed">{s.body}</p>
                </div>
              </FadeIn>
            ))}
          </div>
        </section>

        {/* ═══════════════════════════════════
            SYSTEM STATUS + CTA
            ═══════════════════════════════════ */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-5">
          {/* System Health */}
          <FadeIn delay={0}>
            <div className="h-full p-5 sm:p-6 rounded-xl border border-slate-800/60"
              style={{ background: 'linear-gradient(180deg, #19232e 0%, #151e29 100%)' }}>
              <h3 className="text-lg font-semibold text-white mb-5 flex items-center gap-2">
                <span className="material-icons-outlined text-emerald-400 text-xl">monitor_heart</span>
                System Health
              </h3>
              <div className="space-y-4">
                {[
                  { icon: 'bolt', label: 'Inference Engine', sub: `Latency: <${metrics.inference_ms}ms`, color: '#10b981' },
                  { icon: 'model_training', label: 'Model Version', sub: metrics.model_version, color: '#3b82f6' },
                  { icon: 'science', label: 'Dataset', sub: `${metrics.dataset_patients}+ patients validated`, color: '#10b981' },
                ].map((item) => (
                  <div key={item.label} className="flex items-center justify-between p-3 rounded-lg"
                    style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.04)' }}>
                    <div className="flex items-center gap-3">
                      <div className="w-9 h-9 rounded-lg flex items-center justify-center"
                        style={{ background: `${item.color}15` }}>
                        <span className="material-icons-outlined text-lg" style={{ color: item.color }}>{item.icon}</span>
                      </div>
                      <div>
                        <p className="text-sm font-medium text-white">{item.label}</p>
                        <p className="text-xs text-slate-500">{item.sub}</p>
                      </div>
                    </div>
                    <span className="relative flex h-2.5 w-2.5">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-40" style={{ background: item.color }} />
                      <span className="relative inline-flex rounded-full h-2.5 w-2.5" style={{ background: item.color }} />
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </FadeIn>

          {/* About the Technology Card */}
          <FadeIn delay={0.1} className="lg:col-span-2">
            <div className="h-full rounded-xl relative overflow-hidden flex flex-col justify-center min-h-[240px]"
              style={{ background: 'linear-gradient(180deg, #19232e 0%, #151e29 100%)', border: '1px solid rgba(255,255,255,0.06)' }}>
              {/* Subtle abstract art orbs — very light */}
              <div className="absolute inset-0 pointer-events-none overflow-hidden">
                <div className="absolute top-[-20%] right-[-10%] w-[250px] h-[250px] rounded-full opacity-[0.05]"
                  style={{ background: 'radial-gradient(circle, #137fec 0%, transparent 70%)' }} />
                <div className="absolute bottom-[-30%] left-[10%] w-[200px] h-[200px] rounded-full opacity-[0.03]"
                  style={{ background: 'radial-gradient(circle, #7c3aed 0%, transparent 70%)' }} />
                {/* Subtle grid lines */}
                <div className="absolute inset-0 opacity-[0.015]"
                  style={{ backgroundImage: 'linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)', backgroundSize: '40px 40px' }} />
              </div>
              <div className="relative z-10 p-6 sm:p-8 md:p-10">
                <div className="flex items-center gap-2 mb-5">
                  <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'rgba(19,127,236,0.1)' }}>
                    <span className="material-icons-outlined text-primary" style={{ fontSize: '18px' }}>auto_awesome</span>
                  </div>
                  <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Under the hood</span>
                </div>
                <h2 className="text-xl sm:text-2xl font-bold text-white mb-3 tracking-tight">Multimodal AI for blood diagnostics.</h2>
                <p className="text-slate-400 text-sm max-w-lg mb-6 leading-relaxed">
                  Dual-stream architecture fuses ResNet-50 visual features with 20
                  handcrafted morphological measurements via late fusion. Every prediction includes Grad-CAM explanations.
                </p>
                <div className="flex flex-wrap items-center gap-3">
                  <Link
                    to="/about"
                    className="btn-glass inline-flex items-center gap-2 px-5 py-2.5 text-sm font-semibold"
                  >
                    <span className="material-icons-outlined" style={{ fontSize: '16px' }}>read_more</span>
                    Explore Architecture
                  </Link>
                  <div className="flex items-center gap-4 text-xs text-slate-500">
                    <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block" />ResNet-50</span>
                    <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-violet-400 inline-block" />Grad-CAM</span>
                    <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-cyan-400 inline-block" />Late Fusion</span>
                  </div>
                </div>
              </div>
            </div>
          </FadeIn>
        </div>
      </div>
    </div>
  );
}

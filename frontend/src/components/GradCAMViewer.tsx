import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';

interface GradCAMViewerProps {
  originalImage: string | null;
  gradcamBase64: string | null;
}

const views = [
  { id: 'original', label: 'Original', icon: 'image' },
  { id: 'gradcam', label: 'Grad-CAM', icon: 'thermostat' },
  { id: 'overlay', label: 'Overlay', icon: 'layers' },
] as const;

type ViewId = typeof views[number]['id'];

export default function GradCAMViewer({ originalImage, gradcamBase64 }: GradCAMViewerProps) {
  const [activeView, setActiveView] = useState<ViewId>('overlay');

  const getImageSrc = () => {
    switch (activeView) {
      case 'original':
        return originalImage;
      case 'gradcam':
      case 'overlay':
        return gradcamBase64 ? `data:image/png;base64,${gradcamBase64}` : originalImage;
      default:
        return originalImage;
    }
  };

  const imageSrc = getImageSrc();

  return (
    <div className="rounded-xl border border-slate-800 bg-surface p-8">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <div className="w-9 h-9 rounded-lg bg-primary/10 flex items-center justify-center">
          <span className="material-icons-outlined text-primary text-lg">visibility</span>
        </div>
        <div>
          <h3 className="text-base font-semibold text-white tracking-tight">
            Explainability
          </h3>
          <p className="text-sm text-slate-400">
            Grad-CAM highlights regions that influenced the prediction
          </p>
        </div>
      </div>

      {/* View Selector — segmented control */}
      <div className="flex gap-1 mb-6 p-1 bg-slate-800 rounded-lg inline-flex">
        {views.map((view) => {
          const isActive = activeView === view.id;
          return (
            <button
              key={view.id}
              onClick={() => setActiveView(view.id)}
              className={clsx(
                'flex items-center gap-1.5 px-4 py-2 rounded-md text-sm font-medium transition-all duration-200',
                isActive
                  ? 'bg-primary text-white shadow-glow'
                  : 'text-slate-400 hover:text-white'
              )}
            >
              <span className="material-icons-outlined text-base">{view.icon}</span>
              {view.label}
            </button>
          );
        })}
      </div>

      {/* Image Display */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeView}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
          className="relative aspect-square bg-slate-900 rounded-xl overflow-hidden max-w-md mx-auto border border-slate-800"
        >
          {imageSrc ? (
            <img
              src={imageSrc}
              alt={`${activeView} view`}
              className="w-full h-full object-contain"
            />
          ) : (
            <div className="flex flex-col items-center justify-center h-full gap-2">
              <span className="material-icons-outlined text-3xl text-slate-600">image</span>
              <p className="text-sm text-slate-500">No image available</p>
            </div>
          )}
        </motion.div>
      </AnimatePresence>

      {/* Legend */}
      <div className="mt-6 flex items-center justify-between flex-wrap gap-3">
        <span className="text-xs font-medium text-slate-500">Activation Intensity</span>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500">Low</span>
          <div className="w-24 h-1.5 rounded-full bg-gradient-to-r from-blue-500 via-green-500 via-yellow-400 to-red-500" />
          <span className="text-xs text-slate-500">High</span>
        </div>
      </div>
    </div>
  );
}

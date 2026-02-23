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
    <div className="rounded-xl border border-slate-800 bg-surface overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-6 pt-5 pb-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
            <span className="material-icons-outlined text-primary text-base">visibility</span>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white">Explainability</h3>
            <p className="text-xs text-slate-500">Regions that influenced the prediction</p>
          </div>
        </div>

        {/* View Selector — segmented control */}
        <div className="flex gap-0.5 p-0.5 bg-slate-800/80 rounded-lg">
          {views.map((view) => {
            const isActive = activeView === view.id;
            return (
              <button
                key={view.id}
                onClick={() => setActiveView(view.id)}
                className={clsx(
                  'flex items-center gap-1 px-3 py-1.5 rounded-md text-xs font-medium transition-all duration-200',
                  isActive
                    ? 'bg-primary text-white shadow-md'
                    : 'text-slate-500 hover:text-slate-300'
                )}
              >
                <span className="material-icons-outlined text-sm">{view.icon}</span>
                <span className="hidden sm:inline">{view.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Image Display */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeView}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
          className="relative bg-slate-900 rounded-lg overflow-hidden border border-slate-800 mx-4 mb-4 flex items-center justify-center"
          style={{ minHeight: '220px' }}
        >
          {imageSrc ? (
            <img
              src={imageSrc}
              alt={`${activeView} view`}
              className="max-w-full max-h-[480px] object-contain"
            />
          ) : (
            <div className="flex flex-col items-center justify-center h-full gap-2 py-16">
              <span className="material-icons-outlined text-3xl text-slate-700">image</span>
              <p className="text-xs text-slate-600">No image available</p>
            </div>
          )}
        </motion.div>
      </AnimatePresence>

      {/* Legend */}
      <div className="px-6 pb-4 flex items-center justify-between">
        <span className="text-[10px] font-medium text-slate-600 uppercase tracking-wider">Activation Intensity</span>
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] text-slate-600">Low</span>
          <div className="w-20 h-1 rounded-full bg-gradient-to-r from-blue-500 via-green-500 via-yellow-400 to-red-500" />
          <span className="text-[10px] text-slate-600">High</span>
        </div>
      </div>
    </div>
  );
}

import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion } from 'framer-motion';

interface UploadZoneProps {
  onUpload: (file: File) => void;
}

export default function UploadZone({ onUpload }: UploadZoneProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        onUpload(acceptedFiles[0]);
      }
    },
    [onUpload]
  );

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'],
    },
    multiple: false,
    maxSize: 10 * 1024 * 1024,
  });

  return (
    <div className="space-y-5">
      {/* Heading */}
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'rgba(19,127,236,0.1)' }}>
          <span className="material-icons-outlined text-primary" style={{ fontSize: '18px' }}>add_photo_alternate</span>
        </div>
        <h2 className="text-base font-semibold text-white">Microscopic Imaging</h2>
        <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-md bg-emerald-500/10 text-emerald-400 border border-emerald-500/15">
          Ready
        </span>
      </div>

      {/* Upload Area */}
      <motion.div
        initial={{ opacity: 0, scale: 0.98 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
        {...(getRootProps() as any)}
        className={`
          relative rounded-xl text-center
          cursor-pointer transition-all duration-300 group
          ${isDragActive && !isDragReject
            ? 'border-primary/60 bg-primary/5'
            : isDragReject
            ? 'border-red-500/60 bg-red-500/5'
            : 'hover:bg-primary/[0.03]'
          }
        `}
        style={{
          border: isDragActive && !isDragReject
            ? '1.5px dashed rgba(19,127,236,0.6)'
            : isDragReject
            ? '1.5px dashed rgba(239,68,68,0.6)'
            : '1.5px dashed rgba(255,255,255,0.08)',
          background: isDragActive ? 'rgba(19,127,236,0.03)' : 'rgba(255,255,255,0.01)',
        }}
      >
        <input {...getInputProps()} />

        <div className="py-14 sm:py-20 px-6 space-y-4">
          {/* Icon */}
          <div
            className={`
              mx-auto w-14 h-14 rounded-xl flex items-center justify-center
              transition-all duration-300
              ${isDragActive
                ? 'bg-primary text-white scale-110'
                : isDragReject
                ? 'bg-red-500/15 text-red-400'
                : 'text-slate-400 group-hover:text-primary group-hover:scale-105'
              }
            `}
            style={!isDragActive && !isDragReject ? { background: 'rgba(255,255,255,0.04)' } : {}}
          >
            <span className="material-icons-outlined text-2xl">cloud_upload</span>
          </div>

          {/* Text */}
          <div>
            <h3 className="text-base font-semibold text-white mb-1">
              {isDragActive
                ? 'Drop to upload'
                : isDragReject
                ? 'Invalid file type'
                : 'Drag & Drop blood smear images'}
            </h3>
            <p className="text-sm text-slate-500 max-w-xs mx-auto">
              {isDragReject
                ? 'Please upload a JPG, PNG, or TIFF file'
                : 'Supports JPEG, PNG, or TIFF formats from high-res microscopy.'}
            </p>
          </div>

          {/* Browse button */}
          <button className="px-5 py-2 rounded-lg text-sm font-medium text-slate-300 hover:text-white transition-colors"
            style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.08)' }}>
            Browse Files
          </button>
        </div>
      </motion.div>

      {/* Tip cards */}
      <div className="grid sm:grid-cols-3 gap-3">
        {[
          {
            icon: 'microscope',
            title: 'Best Results',
            desc: 'Single-cell crops or full blood smear fields — auto-segmented',
            color: '#137fec',
          },
          {
            icon: 'grid_view',
            title: 'Multi-Cell Ready',
            desc: 'Upload a whole slide — cells are auto-detected & analyzed individually',
            color: '#8b5cf6',
          },
          {
            icon: 'tune',
            title: 'Image Quality',
            desc: 'Good focus and staining for accurate analysis',
            color: '#10b981',
          },
        ].map((tip) => (
          <div key={tip.title} className="p-3.5 rounded-xl border border-slate-800/60"
            style={{ background: 'linear-gradient(180deg, #19232e 0%, #151e29 100%)' }}>
            <div className="flex items-center gap-2.5 mb-1.5">
              <span className="material-icons-outlined" style={{ fontSize: '16px', color: tip.color }}>{tip.icon}</span>
              <span className="text-sm font-semibold text-white">{tip.title}</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">{tip.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

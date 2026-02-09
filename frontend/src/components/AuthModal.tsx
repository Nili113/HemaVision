import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../contexts/AuthContext';

declare global {
  interface Window {
    google?: {
      accounts: {
        id: {
          initialize: (config: Record<string, unknown>) => void;
          renderButton: (el: HTMLElement, config: Record<string, unknown>) => void;
          prompt: () => void;
        };
      };
    };
  }
}

const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID || '';

interface AuthModalProps {
  open: boolean;
  onClose: () => void;
}

export default function AuthModal({ open, onClose }: AuthModalProps) {
  const { login, register, googleLogin } = useAuth();
  const [tab, setTab] = useState<'login' | 'register'>('login');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const formRef = useRef<HTMLFormElement>(null);
  const googleBtnRef = useRef<HTMLDivElement>(null);

  // Reset on open
  useEffect(() => {
    if (open) {
      setError('');
      setLoading(false);
    }
  }, [open, tab]);

  // Close on Escape key
  useEffect(() => {
    if (!open) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [open, onClose]);

  // Handle Google credential response
  const handleGoogleResponse = useCallback(async (response: { credential: string }) => {
    setError('');
    setLoading(true);
    try {
      await googleLogin(response.credential);
      onClose();
    } catch (err: any) {
      const msg = err?.response?.data?.detail || 'Google sign-in failed';
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [googleLogin, onClose]);

  // Initialize Google Sign-In button
  useEffect(() => {
    if (!open || !GOOGLE_CLIENT_ID) return;

    const timer = setTimeout(() => {
      if (window.google?.accounts?.id && googleBtnRef.current) {
        window.google.accounts.id.initialize({
          client_id: GOOGLE_CLIENT_ID,
          callback: handleGoogleResponse,
        });
        window.google.accounts.id.renderButton(googleBtnRef.current, {
          type: 'standard',
          theme: 'filled_black',
          size: 'large',
          width: '100%',
          text: tab === 'login' ? 'signin_with' : 'signup_with',
          shape: 'pill',
        });
      }
    }, 200);

    return () => clearTimeout(timer);
  }, [open, tab, handleGoogleResponse]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError('');
    setLoading(true);

    const form = formRef.current;
    if (!form) return;

    const fd = new FormData(form);

    try {
      if (tab === 'login') {
        await login({
          username: fd.get('username') as string,
          password: fd.get('password') as string,
        });
      } else {
        await register({
          username: fd.get('username') as string,
          email: fd.get('email') as string,
          password: fd.get('password') as string,
          display_name: fd.get('display_name') as string || undefined,
          sex: fd.get('sex') as string || 'Male',
        });
      }
      onClose();
    } catch (err: any) {
      const msg = err?.response?.data?.detail || 'Something went wrong';
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  return (
    <AnimatePresence>
      {open && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] bg-black/60 backdrop-blur-sm"
            onClick={onClose}
          />
          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ duration: 0.25, ease: [0.25, 0.1, 0.25, 1] }}
            className="fixed inset-0 z-[101] flex items-center justify-center p-4"
            onClick={(e) => e.target === e.currentTarget && onClose()}
          >
            <div
              className="w-full max-w-md rounded-2xl border border-slate-800/80 shadow-2xl relative flex flex-col max-h-[calc(100dvh-2rem)]"
              style={{ background: 'linear-gradient(180deg, #19232e 0%, #151e29 100%)' }}
            >
              {/* Subtle abstract art bg inside modal */}
              <div className="absolute inset-0 pointer-events-none overflow-hidden rounded-2xl">
                <div className="absolute top-[-30%] right-[-20%] w-[300px] h-[300px] rounded-full opacity-[0.06]"
                  style={{ background: 'radial-gradient(circle, #137fec 0%, transparent 70%)' }} />
                <div className="absolute bottom-[-20%] left-[-15%] w-[200px] h-[200px] rounded-full opacity-[0.04]"
                  style={{ background: 'radial-gradient(circle, #a78bfa 0%, transparent 70%)' }} />
              </div>

              {/* Close button — inside modal, top-right */}
              <button
                type="button"
                onClick={onClose}
                className="absolute top-3 right-3 z-20 w-8 h-8 rounded-lg flex items-center justify-center
                  text-slate-500 hover:text-white hover:bg-white/10 transition-all duration-150"
                aria-label="Close"
              >
                <span className="material-icons-outlined" style={{ fontSize: '18px' }}>close</span>
              </button>

              {/* Header with tabs */}
              <div className="relative z-10 px-6 pt-6 pb-4 shrink-0">
                <div className="flex items-center gap-3 mb-5 pr-8">
                  <div className="w-9 h-9 rounded-xl flex items-center justify-center shrink-0"
                    style={{ background: 'linear-gradient(135deg, rgba(19,127,236,0.2) 0%, rgba(124,58,237,0.15) 100%)' }}>
                    <span className="material-icons-outlined text-primary" style={{ fontSize: '20px' }}>person</span>
                  </div>
                  <div className="min-w-0">
                    <h2 className="text-lg font-bold text-white leading-tight">
                      {tab === 'login' ? 'Welcome back' : 'Create account'}
                    </h2>
                    <p className="text-xs text-slate-500">
                      {tab === 'login' ? 'Sign in to your HemaVision account' : 'Join the HemaVision platform'}
                    </p>
                  </div>
                </div>

                {/* Tab switcher */}
                <div className="flex rounded-xl p-1" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
                  {(['login', 'register'] as const).map((t) => (
                    <button
                      key={t}
                      type="button"
                      onClick={() => { setTab(t); setError(''); }}
                      className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all duration-200 ${
                        tab === t
                          ? 'bg-primary/15 text-primary'
                          : 'text-slate-500 hover:text-slate-300'
                      }`}
                    >
                      {t === 'login' ? 'Sign In' : 'Register'}
                    </button>
                  ))}
                </div>
              </div>

              {/* Form — scrollable when content overflows */}
              <form ref={formRef} onSubmit={handleSubmit} className="relative z-10 px-6 pb-6 space-y-4 overflow-y-auto overscroll-contain">
                {error && (
                  <div className="px-3 py-2.5 rounded-lg text-xs font-medium text-red-300 flex items-center gap-2"
                    style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)' }}>
                    <span className="material-icons-outlined" style={{ fontSize: '14px' }}>error</span>
                    {error}
                  </div>
                )}

                {/* Google Sign-In */}
                {GOOGLE_CLIENT_ID ? (
                  <>
                    <div ref={googleBtnRef} className="flex justify-center [&>div]:!w-full" />
                    <div className="flex items-center gap-3">
                      <div className="flex-1 h-px" style={{ background: 'rgba(255,255,255,0.06)' }} />
                      <span className="text-[10px] font-semibold text-slate-600 uppercase tracking-wider">or</span>
                      <div className="flex-1 h-px" style={{ background: 'rgba(255,255,255,0.06)' }} />
                    </div>
                  </>
                ) : (
                  /* Styled Google button (fallback when no client ID) */
                  <>
                    <button
                      type="button"
                      onClick={() => setError('Google Sign-In requires VITE_GOOGLE_CLIENT_ID in .env')}
                      className="w-full flex items-center justify-center gap-2.5 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 hover:bg-white/[0.06]"
                      style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}
                    >
                      <svg width="18" height="18" viewBox="0 0 24 24">
                        <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z" fill="#4285F4"/>
                        <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
                        <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
                        <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
                      </svg>
                      Continue with Google
                    </button>
                    <div className="flex items-center gap-3">
                      <div className="flex-1 h-px" style={{ background: 'rgba(255,255,255,0.06)' }} />
                      <span className="text-[10px] font-semibold text-slate-600 uppercase tracking-wider">or</span>
                      <div className="flex-1 h-px" style={{ background: 'rgba(255,255,255,0.06)' }} />
                    </div>
                  </>
                )}

                <div className="space-y-3">
                  <div>
                    <label className="text-xs font-medium text-slate-400 mb-1.5 block">Username</label>
                    <input
                      name="username"
                      type="text"
                      required
                      autoComplete="username"
                      className="input-field w-full"
                      placeholder="Enter username"
                    />
                  </div>

                  {tab === 'register' && (
                    <>
                      <div>
                        <label className="text-xs font-medium text-slate-400 mb-1.5 block">Email</label>
                        <input
                          name="email"
                          type="email"
                          required
                          autoComplete="email"
                          className="input-field w-full"
                          placeholder="you@example.com"
                        />
                      </div>
                      <div>
                        <label className="text-xs font-medium text-slate-400 mb-1.5 block">Display Name</label>
                        <input
                          name="display_name"
                          type="text"
                          className="input-field w-full"
                          placeholder="How should we call you?"
                        />
                      </div>
                      <div>
                        <label className="text-xs font-medium text-slate-400 mb-1.5 block">Gender</label>
                        <div className="flex gap-2">
                          {(['Male', 'Female'] as const).map((s) => (
                            <label key={s} className="flex-1">
                              <input type="radio" name="sex" value={s} defaultChecked={s === 'Male'} className="sr-only peer" />
                              <div className="peer-checked:border-primary peer-checked:bg-primary/10 peer-checked:text-primary
                                border border-slate-700 rounded-lg py-2.5 text-center text-xs font-semibold text-slate-400
                                cursor-pointer transition-all hover:border-slate-600">
                                <span className="material-icons-outlined block mb-0.5" style={{ fontSize: '20px' }}>
                                  {s === 'Male' ? 'face' : 'face_3'}
                                </span>
                                {s}
                              </div>
                            </label>
                          ))}
                        </div>
                      </div>
                    </>
                  )}

                  <div>
                    <label className="text-xs font-medium text-slate-400 mb-1.5 block">Password</label>
                    <input
                      name="password"
                      type="password"
                      required
                      minLength={6}
                      autoComplete={tab === 'login' ? 'current-password' : 'new-password'}
                      className="input-field w-full"
                      placeholder="••••••••"
                    />
                  </div>
                </div>

                <button
                  type="submit"
                  disabled={loading}
                  className="btn-primary w-full py-2.5 text-sm font-semibold flex items-center justify-center gap-2 disabled:opacity-50"
                >
                  {loading ? (
                    <span className="material-icons-outlined animate-spin" style={{ fontSize: '16px' }}>progress_activity</span>
                  ) : (
                    <span className="material-icons-outlined" style={{ fontSize: '16px' }}>
                      {tab === 'login' ? 'login' : 'person_add'}
                    </span>
                  )}
                  {tab === 'login' ? 'Sign In' : 'Create Account'}
                </button>
              </form>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

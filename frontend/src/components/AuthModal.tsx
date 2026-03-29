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
  const [tab, setTab] = useState<'login' | 'register'>('register');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const formRef = useRef<HTMLFormElement>(null);
  const googleBtnRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (open) {
      setError('');
      setLoading(false);
    }
  }, [open, tab]);

  useEffect(() => {
    if (!open) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [open, onClose]);

  const handleGoogleResponse = useCallback(async (response: { credential: string }) => {
    setError('');
    setLoading(true);
    try {
      await googleLogin(response.credential);
      onClose();
    } catch (err: any) {
      const msg = err?.response?.data?.detail || 'Google sign-in failed. Please try again.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [googleLogin, onClose]);

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
          theme: 'outline',
          size: 'large',
          width: '100%',
          text: tab === 'login' ? 'signin_with' : 'signup_with',
          shape: 'rectangular',
          logo_alignment: 'left'
        });
      }
    }, 300);
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
      const msg = err?.response?.data?.detail || 'Authentication failed. Please check your credentials.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  return (
    <AnimatePresence>
      {open && (
        <>
          {/* Backdrop with extreme blur and dark overlay */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] bg-[#0d1620]/80 backdrop-blur-md"
            onClick={onClose}
          />

          {/* Modal Container */}
          <div className="fixed inset-0 z-[101] flex items-center justify-center p-4 sm:p-6 drop-shadow-2xl pointer-events-none">
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 10 }}
              transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
              className="w-full max-w-4xl flex flex-col md:flex-row rounded-3xl overflow-hidden pointer-events-auto shadow-2xl border border-white/10 bg-surface/90 backdrop-blur-xl"
            >
              {/* Left Side - Visual / Branding */}
              <div className="hidden md:flex md:w-[40%] relative flex-col justify-between p-6 sm:p-10 overflow-hidden bg-gradient-to-br from-primary/20 via-[#101922] to-background">
                {/* Abstract Orbs */}
                <div className="absolute top-[-20%] left-[-20%] w-64 h-64 bg-primary/30 rounded-full blur-[80px]" />
                <div className="absolute bottom-[-10%] right-[-10%] w-48 h-48 bg-purple-500/20 rounded-full blur-[60px]" />
                
                <div className="relative z-10">
                  <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/10 mb-8 backdrop-blur-sm">
                    <span className="w-2 h-2 rounded-full bg-primary" />
                    <span className="text-xs font-semibold tracking-wide text-white uppercase flex items-center gap-1.5"><span className="material-icons-outlined text-[14px]">science</span>HemaVision</span>
                  </div>
                  <h3 className="text-3xl font-bold text-white leading-tight font-display mb-4">
                    {tab === 'login' ? 'Welcome back to your lab.' : 'Start your precision diagnostics.'}
                  </h3>
                  <p className="text-sm text-slate-400 font-medium">
                    {tab === 'login' 
                      ? 'Access your workspace, review past results, and run new inferences securely.'
                      : 'Join thousands of researchers analyzing blood smear morphology in seconds.'}
                  </p>
                </div>

                <div className="relative z-10 mt-12 bg-white/5 border border-white/10 rounded-2xl p-5 backdrop-blur-sm">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-10 h-10 min-w-[40px] min-h-[40px] shrink-0 flex-none rounded-xl bg-primary/20 flex items-center justify-center border border-primary/30 shadow-[0_0_15px_rgba(19,127,236,0.15)]">
                      <span className="material-icons-outlined text-primary text-[20px]">verified_user</span>
                    </div>
                    <div>
                      <p className="text-sm font-bold text-white">HIPAA Compliant</p>
                      <p className="text-xs text-slate-400 mt-0.5">Enterprise-grade security</p>
                    </div>
                  </div>
                  <div className="h-1 w-full bg-white/10 rounded-full overflow-hidden">
                    <div className="h-full bg-primary w-[85%]" />
                  </div>
                </div>
              </div>

              {/* Right Side - Form */}
              <div className="w-full md:w-[60%] p-6 sm:p-10 relative bg-surface-light/30 flex flex-col justify-center min-h-[500px] md:min-h-0">
                <button
                  type="button"
                  onClick={onClose}
                  className="absolute top-6 right-6 w-8 h-8 flex items-center justify-center rounded-full bg-white/5 text-slate-400 hover:text-white hover:bg-white/10 transition-colors ease-out-custom ease-out-custom border border-white/10 active:scale-[0.97]"
                >
                  <span className="material-icons-outlined text-[18px]">close</span>
                </button>

                <div className="max-w-sm mx-auto w-full">
                  <div className="mb-8">
                    <h2 className="text-2xl font-bold text-white font-display">
                      {tab === 'login' ? 'Sign In' : 'Create Account'}
                    </h2>
                    <p className="text-sm text-slate-400 mt-1">
                      {tab === 'login' ? 'Enter your details to proceed.' : 'Fill in your information to get started.'}
                    </p>
                  </div>

                  {/* Tabs */}
                  <div className="flex p-1 bg-black/40 rounded-lg border border-white/5 mb-8">
                    <button
                      onClick={() => { setTab('login'); setError(''); }}
                      className={`flex-1 py-2 text-sm font-semibold rounded-md transition ease-out-custom ${tab === 'login' ? 'bg-primary text-white shadow-lg shadow-primary/20' : 'text-slate-400 hover:text-white'}`}
                    >
                      Login
                    </button>
                    <button
                      onClick={() => { setTab('register'); setError(''); }}
                      className={`flex-1 py-2 text-sm font-semibold rounded-md transition ease-out-custom ${tab === 'register' ? 'bg-primary text-white shadow-lg shadow-primary/20' : 'text-slate-400 hover:text-white'}`}
                    >
                      Register
                    </button>
                  </div>

                  <form ref={formRef} onSubmit={handleSubmit} className="space-y-4">
                    {/* Error Banner */}
                    {error && (
                      <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} className="overflow-hidden">
                        <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center gap-2 text-red-400 text-xs font-medium">
                          <span className="material-icons-outlined text-[16px]">error_outline</span>
                          {error}
                        </div>
                      </motion.div>
                    )}

                    {/* Google Auth Block */}
                    {GOOGLE_CLIENT_ID ? (
                      <div className="mb-6">
                        <div ref={googleBtnRef} className="w-full flex justify-center [&>div]:!w-full overflow-hidden rounded-lg border border-white/10" />
                        <div className="flex items-center gap-4 mt-6 mb-2">
                          <div className="flex-1 h-px bg-white/10" />
                          <span className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Or continue with email</span>
                          <div className="flex-1 h-px bg-white/10" />
                        </div>
                      </div>
                    ) : null}

                    {/* Form Fields */}
                    <div className="space-y-4">
                      {tab === 'register' && (
                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-1.5 focus-within:text-primary transition-colors ease-out-custom ease-out-custom text-slate-400">
                            <label className="text-xs font-semibold tracking-wide uppercase">Display Name</label>
                            <input name="display_name" type="text" className="w-full px-4 py-3 bg-black/40 border border-white/10 rounded-lg text-white text-sm focus:border-primary focus:ring-1 focus:ring-primary outline-none transition ease-out-custom placeholder:text-slate-600" placeholder="Dr. Smith" />
                          </div>
                          <div className="space-y-1.5 focus-within:text-primary transition-colors ease-out-custom ease-out-custom text-slate-400">
                            <label className="text-xs font-semibold tracking-wide uppercase">Gender</label>
                            <select name="sex" className="w-full px-4 py-3 bg-black/40 border border-white/10 rounded-lg text-white text-sm focus:border-primary focus:ring-1 focus:ring-primary outline-none transition ease-out-custom appearance-none">
                              <option value="Male">Male</option>
                              <option value="Female">Female</option>
                            </select>
                          </div>
                        </div>
                      )}
                      
                      <div className="space-y-1.5 focus-within:text-primary transition-colors ease-out-custom ease-out-custom text-slate-400">
                        <label className="text-xs font-semibold tracking-wide uppercase">Username <span className="text-red-500">*</span></label>
                        <input name="username" type="text" required className="w-full px-4 py-3 bg-black/40 border border-white/10 rounded-lg text-white text-sm focus:border-primary focus:ring-1 focus:ring-primary outline-none transition ease-out-custom placeholder:text-slate-600" placeholder="Unique username" />
                      </div>

                      {tab === 'register' && (
                        <div className="space-y-1.5 focus-within:text-primary transition-colors ease-out-custom ease-out-custom text-slate-400">
                          <label className="text-xs font-semibold tracking-wide uppercase">Email Address <span className="text-red-500">*</span></label>
                          <input name="email" type="email" required className="w-full px-4 py-3 bg-black/40 border border-white/10 rounded-lg text-white text-sm focus:border-primary focus:ring-1 focus:ring-primary outline-none transition ease-out-custom placeholder:text-slate-600" placeholder="researcher@hospital.org" />
                        </div>
                      )}

                      <div className="space-y-1.5 focus-within:text-primary transition-colors ease-out-custom ease-out-custom text-slate-400">
                        <label className="text-xs font-semibold tracking-wide uppercase">Password <span className="text-red-500">*</span></label>
                        <input name="password" type="password" required minLength={6} className="w-full px-4 py-3 bg-black/40 border border-white/10 rounded-lg text-white text-sm focus:border-primary focus:ring-1 focus:ring-primary outline-none transition ease-out-custom placeholder:text-slate-600" placeholder="••••••••" />
                      </div>
                    </div>

                    <button
                      type="submit"
                      disabled={loading}
                      className="w-full py-3.5 mt-6 bg-primary hover:bg-primary-hover text-white rounded-lg font-semibold text-sm transition ease-out-custom flex items-center justify-center gap-2 disabled:opacity-70 disabled:cursor-not-allowed shadow-[0_0_20px_rgba(19,127,236,0.3)] hover:shadow-[0_0_30px_rgba(19,127,236,0.5)] border border-primary-300 active:scale-[0.97]"
                    >
                      {loading ? (
                        <>
                          <span className="material-icons-outlined animate-spin text-[18px]">autorenew</span>
                          Processing...
                        </>
                      ) : (
                        <>
                          {tab === 'login' ? 'Secure Sign In' : 'Create Account'}
                          <span className="material-icons-outlined text-[18px]">arrow_forward</span>
                        </>
                      )}
                    </button>
                  </form>
                </div>
              </div>
            </motion.div>
          </div>
        </>
      )}
    </AnimatePresence>
  );
}

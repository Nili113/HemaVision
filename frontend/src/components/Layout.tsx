import { useState, useEffect, useRef } from 'react';
import { Link, Outlet, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Menu, X } from 'lucide-react';
import clsx from 'clsx';
import { useAuth } from '../contexts/AuthContext';
import AuthModal from './AuthModal';

const navItems = [
  { path: '/', label: 'Dashboard', icon: 'dashboard' },
  { path: '/analyze', label: 'Analyze', icon: 'biotech' },
  { path: '/history', label: 'History', icon: 'history' },
  { path: '/about', label: 'About', icon: 'info' },
];

/* ── Gender-based avatar SVGs ────────── */
function MaleAvatar({ size = 32 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="20" cy="20" r="20" fill="url(#maleGrad)" />
      <defs>
        <linearGradient id="maleGrad" x1="0" y1="0" x2="40" y2="40">
          <stop offset="0%" stopColor="#137fec" />
          <stop offset="100%" stopColor="#0e6adb" />
        </linearGradient>
      </defs>
      {/* Head */}
      <circle cx="20" cy="15" r="7" fill="#e2e8f0" />
      {/* Body */}
      <path d="M8 35 C8 26 14 22 20 22 C26 22 32 26 32 35" fill="#e2e8f0" />
      {/* Short hair */}
      <path d="M13 14 C13 9 17 6 20 6 C23 6 27 9 27 14 C27 11 24 9 20 9 C16 9 13 11 13 14Z" fill="#64748b" />
    </svg>
  );
}

function FemaleAvatar({ size = 32 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="20" cy="20" r="20" fill="url(#femaleGrad)" />
      <defs>
        <linearGradient id="femaleGrad" x1="0" y1="0" x2="40" y2="40">
          <stop offset="0%" stopColor="#a78bfa" />
          <stop offset="100%" stopColor="#7c3aed" />
        </linearGradient>
      </defs>
      {/* Head */}
      <circle cx="20" cy="15" r="7" fill="#e2e8f0" />
      {/* Body */}
      <path d="M8 35 C8 26 14 22 20 22 C26 22 32 26 32 35" fill="#e2e8f0" />
      {/* Long hair */}
      <path d="M11 16 C11 8 15 5 20 5 C25 5 29 8 29 16 C29 12 26 8 20 8 C14 8 11 12 11 16Z" fill="#64748b" />
      <path d="M11 16 C10 20 10 22 11 24" stroke="#64748b" strokeWidth="2" strokeLinecap="round" fill="none" />
      <path d="M29 16 C30 20 30 22 29 24" stroke="#64748b" strokeWidth="2" strokeLinecap="round" fill="none" />
    </svg>
  );
}

function UserAvatar({ sex, size = 32 }: { sex: string; size?: number }) {
  return sex === 'Female' ? <FemaleAvatar size={size} /> : <MaleAvatar size={size} />;
}

export default function Layout() {
  const location = useLocation();
  const { user, isAuthenticated, logout } = useAuth();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close mobile menu on route change
  useEffect(() => {
    setMobileMenuOpen(false);
  }, [location.pathname]);

  // Track scroll for navbar shadow
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  // Close dropdown on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setDropdownOpen(false);
      }
    }
    if (dropdownOpen) document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [dropdownOpen]);

  return (
    <div className="min-h-screen bg-background flex flex-col font-display">
      {/* ── Navbar ──────────────────────────────────── */}
      <nav className={clsx(
        'sticky top-0 z-50 transition-all duration-500',
        scrolled
          ? 'bg-[rgba(16,25,34,0.55)] backdrop-blur-2xl border-b border-white/[0.06] shadow-[0_4px_30px_rgba(0,0,0,0.3)]'
          : 'bg-[rgba(16,25,34,0.3)] backdrop-blur-xl border-b border-white/[0.03]'
      )}>
        <div className="section-container">
          <div className="flex items-center justify-between h-14">
            {/* Logo */}
            <Link to="/" className="flex items-center gap-2.5 select-none group">
              <div className="w-7 h-7 rounded-lg flex items-center justify-center transition-transform duration-300 group-hover:scale-110"
                style={{ background: 'linear-gradient(135deg, rgba(19,127,236,0.2) 0%, rgba(124,58,237,0.15) 100%)' }}>
                <span className="material-icons-outlined text-primary" style={{ fontSize: '18px' }}>bubble_chart</span>
              </div>
              <span className="text-base font-bold tracking-tight text-white">
                Hema<span className="text-primary">Vision</span>
              </span>
              <span className="px-1.5 py-0.5 rounded text-[9px] font-semibold text-slate-500 hidden sm:inline"
                style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.06)' }}>
                v1.0
              </span>
            </Link>

            {/* Desktop Nav */}
            <div className="hidden md:flex items-center gap-0.5">
              {navItems.map((item) => {
                const isActive = location.pathname === item.path;
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={clsx(
                      'relative px-3.5 py-1.5 text-[13px] font-medium transition-colors duration-200',
                      isActive
                        ? 'text-white'
                        : 'text-slate-400 hover:text-slate-200'
                    )}
                  >
                    {item.label}
                    {isActive && (
                      <motion.div
                        layoutId="nav-underline"
                        className="absolute bottom-0 left-3.5 right-3.5 h-[2px] rounded-full bg-primary"
                        transition={{ type: 'spring', stiffness: 500, damping: 35 }}
                      />
                    )}
                  </Link>
                );
              })}
            </div>

            {/* User avatar / Sign In + mobile toggle */}
            <div className="flex items-center gap-2">
              {isAuthenticated && user ? (
                /* ── Logged-in: Avatar with dropdown ── */
                <div className="relative" ref={dropdownRef}>
                  <button
                    onClick={() => setDropdownOpen(!dropdownOpen)}
                    className="flex items-center gap-2 rounded-full p-0.5 transition-all duration-200 hover:ring-2 hover:ring-primary/30"
                  >
                    <UserAvatar sex={user.sex} size={32} />
                    <span className="hidden lg:block text-xs font-medium text-slate-300 max-w-[100px] truncate">
                      {user.display_name}
                    </span>
                  </button>

                  <AnimatePresence>
                    {dropdownOpen && (
                      <motion.div
                        initial={{ opacity: 0, y: -6, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: -6, scale: 0.95 }}
                        transition={{ duration: 0.15 }}
                        className="absolute right-0 top-11 w-56 rounded-xl border border-slate-800/80 shadow-2xl overflow-hidden z-50"
                        style={{ background: 'linear-gradient(180deg, #1d2a38 0%, #19232e 100%)' }}
                      >
                        {/* User info */}
                        <div className="px-4 py-3 border-b border-slate-800/60">
                          <div className="flex items-center gap-3">
                            <UserAvatar sex={user.sex} size={36} />
                            <div className="min-w-0">
                              <p className="text-sm font-semibold text-white truncate">{user.display_name}</p>
                              <p className="text-[11px] text-slate-500 truncate">{user.email}</p>
                            </div>
                          </div>
                        </div>

                        {/* Menu items */}
                        <div className="p-1.5">
                          <Link
                            to="/analyze"
                            onClick={() => setDropdownOpen(false)}
                            className="flex items-center gap-2.5 px-3 py-2 text-xs font-medium text-slate-300 hover:text-white hover:bg-white/[0.04] rounded-lg transition-colors"
                          >
                            <span className="material-icons-outlined text-primary" style={{ fontSize: '16px' }}>add_circle</span>
                            New Analysis
                          </Link>
                          <Link
                            to="/history"
                            onClick={() => setDropdownOpen(false)}
                            className="flex items-center gap-2.5 px-3 py-2 text-xs font-medium text-slate-300 hover:text-white hover:bg-white/[0.04] rounded-lg transition-colors"
                          >
                            <span className="material-icons-outlined text-slate-500" style={{ fontSize: '16px' }}>history</span>
                            My History
                          </Link>
                          <div className="my-1 border-t border-slate-800/60" />
                          <button
                            onClick={() => { logout(); setDropdownOpen(false); }}
                            className="w-full flex items-center gap-2.5 px-3 py-2 text-xs font-medium text-red-400 hover:text-red-300 hover:bg-red-500/[0.06] rounded-lg transition-colors"
                          >
                            <span className="material-icons-outlined" style={{ fontSize: '16px' }}>logout</span>
                            Sign Out
                          </button>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              ) : (
                /* ── Not logged in: Sign In button ── */
                <button
                  onClick={() => setAuthModalOpen(true)}
                  className="hidden md:inline-flex items-center gap-1.5 px-4 py-1.5 text-[13px] font-semibold rounded-lg transition-all duration-200 text-slate-300 hover:text-white"
                  style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}
                >
                  <span className="material-icons-outlined" style={{ fontSize: '16px' }}>person</span>
                  Sign In
                </button>
              )}

              <button
                className="md:hidden w-8 h-8 flex items-center justify-center rounded-lg text-slate-400 hover:text-white transition-colors"
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                aria-label="Toggle menu"
              >
                <AnimatePresence mode="wait">
                  {mobileMenuOpen ? (
                    <motion.div key="close" initial={{ rotate: -90, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }} exit={{ rotate: 90, opacity: 0 }} transition={{ duration: 0.15 }}>
                      <X size={18} />
                    </motion.div>
                  ) : (
                    <motion.div key="menu" initial={{ rotate: 90, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }} exit={{ rotate: -90, opacity: 0 }} transition={{ duration: 0.15 }}>
                      <Menu size={18} />
                    </motion.div>
                  )}
                </AnimatePresence>
              </button>
            </div>
          </div>
        </div>

        {/* Mobile drawer */}
        <AnimatePresence>
          {mobileMenuOpen && (
            <>
              {/* Backdrop */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="md:hidden fixed inset-0 top-14 bg-black/40 backdrop-blur-sm z-40"
                onClick={() => setMobileMenuOpen(false)}
              />
              {/* Panel */}
              <motion.div
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.2, ease: [0.25, 0.1, 0.25, 1] }}
                className="md:hidden absolute top-14 left-0 right-0 z-50 border-b border-slate-800/80"
                style={{ background: 'rgba(16,25,34,0.97)', backdropFilter: 'blur(20px)' }}
              >
                <div className="px-4 py-4 space-y-1">
                  {/* User section in mobile */}
                  {isAuthenticated && user ? (
                    <div className="flex items-center gap-3 px-3 py-3 mb-2 rounded-xl" style={{ background: 'rgba(255,255,255,0.02)' }}>
                      <UserAvatar sex={user.sex} size={36} />
                      <div className="min-w-0">
                        <p className="text-sm font-semibold text-white truncate">{user.display_name}</p>
                        <p className="text-[11px] text-slate-500 truncate">{user.email}</p>
                      </div>
                    </div>
                  ) : (
                    <motion.div
                      initial={{ opacity: 0, x: -12 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0, duration: 0.2 }}
                      className="mb-2"
                    >
                      <button
                        onClick={() => { setMobileMenuOpen(false); setAuthModalOpen(true); }}
                        className="flex items-center gap-3 px-3 py-3 rounded-xl text-sm font-medium text-primary w-full"
                        style={{ background: 'rgba(19,127,236,0.08)' }}
                      >
                        <span className="material-icons-outlined" style={{ fontSize: '20px' }}>person</span>
                        Sign In / Register
                      </button>
                    </motion.div>
                  )}

                  {navItems.map((item, i) => {
                    const isActive = location.pathname === item.path;
                    return (
                      <motion.div
                        key={item.path}
                        initial={{ opacity: 0, x: -12 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.04, duration: 0.2 }}
                      >
                        <Link
                          to={item.path}
                          className={clsx(
                            'flex items-center gap-3 px-3 py-3 rounded-xl text-sm font-medium transition-all',
                            isActive
                              ? 'text-white bg-primary/10'
                              : 'text-slate-400 hover:text-white active:bg-slate-800/50'
                          )}
                        >
                          <span className={clsx('material-icons-outlined', isActive ? 'text-primary' : 'text-slate-600')} style={{ fontSize: '20px' }}>
                            {item.icon}
                          </span>
                          {item.label}
                          {isActive && (
                            <div className="ml-auto w-1.5 h-1.5 rounded-full bg-primary" />
                          )}
                        </Link>
                      </motion.div>
                    );
                  })}

                  {isAuthenticated && (
                    <motion.div
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.18, duration: 0.2 }}
                      className="pt-3 border-t border-slate-800/60"
                    >
                      <button
                        onClick={() => { logout(); setMobileMenuOpen(false); }}
                        className="flex items-center justify-center gap-2 w-full py-2.5 rounded-xl text-sm font-semibold text-red-400 hover:bg-red-500/10 transition-colors"
                      >
                        <span className="material-icons-outlined" style={{ fontSize: '18px' }}>logout</span>
                        Sign Out
                      </button>
                    </motion.div>
                  )}
                </div>
              </motion.div>
            </>
          )}
        </AnimatePresence>
      </nav>

      {/* Auth Modal */}
      <AuthModal open={authModalOpen} onClose={() => setAuthModalOpen(false)} />

      {/* ── Page Content ───────────────────────────── */}
      <main className="flex-1">
        <Outlet />
      </main>

      {/* ── Footer ─────────────────────────────────── */}
      <footer className="mt-auto border-t border-slate-800/60 py-5">
        <div className="section-container flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-1.5 text-slate-500 text-xs">
            <span className="material-icons-outlined" style={{ fontSize: '14px' }}>lock</span>
            <span>Research Use Only</span>
          </div>
          <span className="text-xs text-slate-600">
            Built by{' '}
            <span className="text-slate-400 font-medium">Firoj</span>,{' '}
            <span className="text-slate-400 font-medium">Nilima</span> &amp;{' '}
            <span className="text-slate-400 font-medium">Aashika</span>
          </span>
        </div>
      </footer>
    </div>
  );
}

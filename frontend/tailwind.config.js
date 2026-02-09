/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#137fec',
          hover: '#0e6adb',
          50: '#e8f4fd',
          100: '#b8dcfa',
          200: '#88c4f7',
          300: '#4da5f3',
          400: '#1e90ef',
          500: '#137fec',
          600: '#0e6adb',
          700: '#0a50a8',
        },
        background: '#101922',
        surface: {
          DEFAULT: '#19232e',
          light: '#1e2d3d',
          dark: '#0d1620',
        },
      },
      fontFamily: {
        display: ['Space Grotesk', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'SF Mono', 'Fira Code', 'monospace'],
      },
      fontSize: {
        'hero': ['clamp(2.5rem, 5vw, 3.5rem)', { lineHeight: '1.1', letterSpacing: '-0.03em', fontWeight: '700' }],
        'display': ['clamp(1.75rem, 3vw, 2.5rem)', { lineHeight: '1.15', letterSpacing: '-0.025em', fontWeight: '700' }],
        'title': ['clamp(1.25rem, 2vw, 1.5rem)', { lineHeight: '1.25', letterSpacing: '-0.02em', fontWeight: '600' }],
      },
      boxShadow: {
        'glow': '0 0 20px rgba(19, 127, 236, 0.15)',
        'glow-lg': '0 4px 40px rgba(19, 127, 236, 0.25)',
        'card': '0 1px 3px rgba(0,0,0,0.3)',
        'card-hover': '0 8px 30px rgba(0,0,0,0.4)',
        'elevated': '0 16px 48px rgba(0,0,0,0.45)',
      },
      borderRadius: {
        DEFAULT: '0.25rem',
        lg: '0.5rem',
        xl: '0.75rem',
      },
      spacing: {
        '18': '4.5rem',
        '22': '5.5rem',
      },
      animation: {
        'fade-in': 'fadeIn 0.6s cubic-bezier(0.25, 0.1, 0.25, 1)',
        'slide-up': 'slideUp 0.7s cubic-bezier(0.25, 0.1, 0.25, 1)',
        'breathe': 'breathe 4s ease-in-out infinite',
        'glow-pulse': 'glowPulse 2s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(24px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        breathe: {
          '0%, 100%': { opacity: '0.4' },
          '50%': { opacity: '0.8' },
        },
        glowPulse: {
          '0%, 100%': { boxShadow: '0 0 20px rgba(19, 127, 236, 0.15)' },
          '50%': { boxShadow: '0 0 40px rgba(19, 127, 236, 0.3)' },
        },
      },
    },
  },
  plugins: [],
};

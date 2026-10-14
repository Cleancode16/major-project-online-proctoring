/** @type {import('tailwindcss').Config} */
// UI3 design system (flow.txt PART II-D "UI3"): palette scales built around the brand accent #6485E8 and the
// lavender #9B8AFB; semantic surfaces / text colours are CSS variables in src/index.css (light + dark).
export default {
  // the theme toggle in App.jsx adds/removes the "dark" class on <html>
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['-apple-system', 'BlinkMacSystemFont', '"SF Pro Text"', '"SF Pro Display"', '"Segoe UI Variable"',
          '"Segoe UI"', 'Inter', 'Roboto', '"Helvetica Neue"', 'Arial', 'sans-serif'],
        mono: ['"SF Mono"', 'ui-monospace', '"Cascadia Code"', 'Menlo', 'Consolas', 'monospace'],
      },
      colors: {
        // 500 = brand accent; 600+ carry white text at >= 4.5:1
        primary: {
          50: '#F2F5FE', 100: '#E4EAFC', 200: '#C9D5F8', 300: '#A5B8F3', 400: '#829DEE',
          500: '#6485E8', 600: '#4F6FD8', 700: '#3F59B8', 800: '#344A93', 900: '#2D3F76', 950: '#1C264A',
        },
        accent: {
          50: '#F5F3FF', 100: '#ECE8FE', 200: '#DAD3FD', 300: '#C2B6FC', 400: '#AE9FFB',
          500: '#9B8AFB', 600: '#6E5CE6', 700: '#5A48C9', 800: '#4A3BA3', 900: '#3D3282',
        },
        ink: { DEFAULT: '#172033', muted: '#5B6579', subtle: '#646F86' },
      },
      borderRadius: {
        '2xl': '1.25rem',
        '3xl': '1.5rem',
      },
      boxShadow: {
        glass: '0 1px 1px rgba(23, 32, 51, 0.04), 0 8px 24px -6px rgba(23, 32, 51, 0.10)',
        'glass-lg': '0 2px 4px rgba(23, 32, 51, 0.04), 0 18px 40px -12px rgba(23, 32, 51, 0.18)',
        glow: '0 6px 18px -6px rgba(79, 111, 216, 0.55)',
      },
      keyframes: {
        shimmer: { '100%': { transform: 'translateX(100%)' } },
        'fade-in': { from: { opacity: '0' }, to: { opacity: '1' } },
        'slide-up': { from: { opacity: '0', transform: 'translateY(8px)' }, to: { opacity: '1', transform: 'none' } },
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        shimmer: 'shimmer 1.6s ease-in-out infinite',
        'fade-in': 'fade-in 200ms ease-out both',
        'slide-up': 'slide-up 240ms cubic-bezier(0.2, 0.8, 0.2, 1) both',
      },
    },
  },
  plugins: [],
}

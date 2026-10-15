/** @type {import('tailwindcss').Config} */
// UI4 design system: every colour is a CSS variable defined in src/index.css (light "Warm Neutral" + dark
// "Graphite"), so utilities such as bg-surface / text-fg-muted / border-line follow the theme with no dark: variants.
const v = (name) => `var(--${name})`;

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
        sans: ['Inter', '"Segoe UI Variable"', '"Segoe UI"', '-apple-system', 'BlinkMacSystemFont', '"SF Pro Text"',
          'Roboto', '"Helvetica Neue"', 'Arial', 'sans-serif'],
        mono: ['"SF Mono"', 'ui-monospace', '"Cascadia Code"', 'Menlo', 'Consolas', 'monospace'],
      },
      colors: {
        canvas: v('bg'),
        sidebar: v('sidebar'),
        surface: v('surface'),
        hover: v('hover'),
        line: v('border'),
        fg: { DEFAULT: v('text'), muted: v('text-muted'), subtle: v('text-subtle') },
        accent: { DEFAULT: v('accent'), hover: v('accent-hover'), fg: v('on-accent'), soft: v('accent-soft') },
        success: { DEFAULT: v('success'), fg: v('success-fg'), soft: v('success-soft') },
        warning: { DEFAULT: v('warning'), fg: v('warning-fg'), soft: v('warning-soft') },
        critical: { DEFAULT: v('critical'), fg: v('critical-fg'), soft: v('critical-soft') },
        info: { DEFAULT: v('info'), fg: v('info-fg'), soft: v('info-soft') },
      },
      fontSize: {
        '2xs': ['0.6875rem', { lineHeight: '1rem' }],     // 11px: badges, axis labels
      },
      // Radius scale clamped to 4-10 px: legacy rounded-xl / -2xl / -3xl classes resolve to the same quiet radii
      borderRadius: {
        sm: '4px',
        DEFAULT: '6px',
        md: '6px',
        lg: '8px',
        xl: '8px',
        '2xl': '8px',
        '3xl': '10px',
      },
      boxShadow: {
        popover: '0 4px 16px -4px rgba(0, 0, 0, 0.14), 0 1px 2px rgba(0, 0, 0, 0.06)',
      },
      keyframes: {
        shimmer: { '100%': { transform: 'translateX(100%)' } },
        'fade-in': { from: { opacity: '0' }, to: { opacity: '1' } },
      },
      animation: {
        shimmer: 'shimmer 1.6s ease-in-out infinite',
        'fade-in': 'fade-in 150ms ease-out both',
      },
    },
  },
  plugins: [],
}

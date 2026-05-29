/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'neon-cyan': '#00ffcc',
        'neon-blue': '#00ccff',
        'neon-purple': '#cc00ff',
        'dark-bg': '#0a0a0f',
      },
      backdropBlur: {
        'xs': '2px',
      },
    },
  },
  plugins: [],
}

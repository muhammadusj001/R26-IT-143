/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        pool: {
          50: '#eefbf8',
          100: '#d7f4ee',
          200: '#afe9de',
          300: '#7edac8',
          400: '#48c4ae',
          500: '#24a994',
          600: '#1b8779',
          700: '#196e63',
          800: '#175850',
          900: '#154845',
        },
      },
    },
  },
  plugins: [],
};

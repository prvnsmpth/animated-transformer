/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{html,js,svelte,ts}'],
  theme: {
    extend: {},
    fontFamily: {
      headline: ["Canela", "Times\\ New\\ Roman", "serif"],
      subheadline: 'AtlasGrotesk, Helvetica, Arial, sans-serif',
      para: 'Publico'
    }
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}


<script lang="ts">
  import '../app.css'

  import { MetaTags } from 'svelte-meta-tags'
</script>

<svelte:head>
  <title>The Animated Transformer</title>
  <script defer data-domain="prvnsmpth.github.io" src="https://plausible.io/js/script.js"></script>
</svelte:head>

<MetaTags
  title="The Animated Transformer"
  description="An intuitive explanation of the Transformer model architecture."
  openGraph={{
    url: 'https://prvnsmpth.github.io/animated-transformer',
    title: 'The Animated Transformer',
    description: 'An intuitive explanation of the Transformer model architecture.',
    images: [
      {
        url: 'https://prvnsmpth.github.io/animated-transformer/og.png',
        width: 1920,
        height: 1080,
        alt: 'The Animated Transformer'
      }
    ]
  }}
/>

<slot />

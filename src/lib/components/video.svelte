<script lang="ts">
  export let src: string

  let video: HTMLVideoElement

  // These values are bound to properties of the video
  let time = 0
  let duration: number
  let paused = true

  let showControls = true
  let showControlsTimeout: number

  // Used to track time of last mouse down event
  let lastMouseDown: Date

  function handleMove(e: MouseEvent | TouchEvent) {
    // Make the controls visible, but fade out after
    // 2.5 seconds of inactivity
    clearTimeout(showControlsTimeout)
    showControlsTimeout = setTimeout(() => (showControls = false), 2500)
    showControls = true

    if (!duration) return // video not loaded yet
    if (e.type !== 'touchmove') {
      e = e as MouseEvent
      if (!(e.buttons & 1)) return // mouse not down
    }

    const clientX = e.type === 'touchmove' ? (e as TouchEvent).touches[0].clientX : (e as MouseEvent).clientX
    const { left, right } = video.getBoundingClientRect()
    time = (duration * (clientX - left)) / (right - left)
  }

  // we can't rely on the built-in click event, because it fires
  // after a drag — we have to listen for clicks ourselves
  function handleMousedown(e: MouseEvent) {
    lastMouseDown = new Date()
  }

  function handleMouseup(e: MouseEvent) {
    if (new Date().getTime() - lastMouseDown.getTime() < 300) {
      if (e.target) {
        const el = e.target as HTMLVideoElement
        if (paused) el.play()
        else el.pause()
      }
    }
  }
</script>

<video
  bind:this={video}
  {src}
  on:mousemove={handleMove}
  on:touchmove|preventDefault={handleMove}
  on:mousedown={handleMousedown}
  on:mouseup={handleMouseup}
  bind:currentTime={time}
  bind:duration
  bind:paused
  autoplay muted loop playsinline
>
  <track kind="captions" />
</video>

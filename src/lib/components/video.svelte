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

    const clientX =
      e.type === 'touchmove' ? (e as TouchEvent).touches[0].clientX : (e as MouseEvent).clientX
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

<div class="relative">
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
    autoplay
    muted
    loop
    playsinline
  >
    <track kind="captions" />
  </video>

  {#if paused}
    <div class="absolute z-10 top-0 left-0">
      <svg
        height="36px"
        style="enable-background:new 0 0 512 512;"
        viewBox="0 0 512 512"
        width="36px"
      >
        <g>
          <path
            d="M224,435.8V76.1c0-6.7-5.4-12.1-12.2-12.1h-71.6c-6.8,0-12.2,5.4-12.2,12.1v359.7c0,6.7,5.4,12.2,12.2,12.2h71.6   C218.6,448,224,442.6,224,435.8z"
          />
          <path
            d="M371.8,64h-71.6c-6.7,0-12.2,5.4-12.2,12.1v359.7c0,6.7,5.4,12.2,12.2,12.2h71.6c6.7,0,12.2-5.4,12.2-12.2V76.1   C384,69.4,378.6,64,371.8,64z"
          />
        </g>
      </svg>
    </div>
  {/if}
</div>

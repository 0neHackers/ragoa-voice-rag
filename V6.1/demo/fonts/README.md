# Drop-in fonts

## Disket Mono

Disket Mono isn't on Google Fonts, npm, or any package CDN — Fontfabric releases it
directly and doesn't redistribute it. So it can't be fetched at build time the way the
other two faces are.

To use it, put these two files here:

```
demo/fonts/DisketMono-Regular.woff2
demo/fonts/DisketMono-Bold.woff2
```

The page already declares the `@font-face` rules pointing at `/fonts/…`, so they'll be
picked up on the next reload with no code change.

If you only have `.ttf` or `.otf`, convert them first — woff2 is roughly a third of the
size, which matters on a first paint:

```bash
pip install fonttools brotli
fonttools ttLib.woff2 compress DisketMono-Regular.ttf
```

**Without these files nothing breaks.** The stack falls back to Space Mono, which has the
same squared-off terminal character. That's a deliberate fallback rather than the browser
quietly dropping to Arial and wrecking the type system.

## The other faces

Loaded from CDNs, no action needed:

- **JetBrains Mono** — Google Fonts. Body text, data, numbers.
- **Cal Sans** — jsDelivr (`@fontsource/cal-sans`). Display headings only.
- **Noto Sans Devanagari** — Google Fonts. This one isn't optional: neither JetBrains Mono
  nor Cal Sans has Devanagari glyphs, and the entire corpus and every answer is Hindi.
  Without it the actual content renders in whatever the OS falls back to.

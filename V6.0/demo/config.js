// Points the UI at its backend.
//
// Leave this alone when FastAPI serves the page (the normal setup) — same-origin
// requests just work. Only set it if you host the frontend somewhere else, e.g. the
// static page on Vercel with the pipeline on Render:
//
//   window.__API_BASE__ = 'https://voice-rag-0nehackers.onrender.com';
//
// No trailing slash needed; it gets stripped either way.
window.__API_BASE__ = '';

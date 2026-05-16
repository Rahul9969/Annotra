/** API origin — override with VITE_API_URL when frontend is hosted separately (Vercel, Netlify). */
export function apiBase(): string {
  const raw = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8765';
  return raw.replace(/\/$/, '');
}

export function wsBase(): string {
  return apiBase().replace(/^https:/, 'wss:').replace(/^http:/, 'ws:');
}

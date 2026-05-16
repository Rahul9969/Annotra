/** API origin — override with VITE_API_URL when frontend is hosted separately (Vercel, Netlify). */
export function apiBase(): string {
  const raw = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8765';
  return raw.replace(/\/$/, '');
}

/** Production build without VITE_API_URL still points at localhost — cloud UI cannot reach the API. */
export function apiMisconfiguredForCloud(): boolean {
  return import.meta.env.PROD && !import.meta.env.VITE_API_URL;
}

export function backendOfflineHint(): string {
  if (apiMisconfiguredForCloud()) {
    return 'Cloud UI misconfigured — set VITE_API_URL to your Render API URL on Vercel, then redeploy';
  }
  return 'Backend offline — run: cd marine-annotation-studio/backend && python -m uvicorn app.main:app --port 8765';
}

export function wsBase(): string {
  return apiBase().replace(/^https:/, 'wss:').replace(/^http:/, 'ws:');
}

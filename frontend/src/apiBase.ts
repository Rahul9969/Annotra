/** Default API when VITE_API_URL is unset (local dev, Electron, vite preview on localhost). */
const DEFAULT_LOCAL_API = 'http://127.0.0.1:8765';

/** API origin — set VITE_API_URL on Vercel/Netlify; leave unset for local/Electron. */
export function apiBase(): string {
  const raw = import.meta.env.VITE_API_URL || DEFAULT_LOCAL_API;
  return String(raw).replace(/\/$/, '');
}

function browserHostname(): string {
  if (typeof window === 'undefined') return '';
  return window.location.hostname;
}

export function isElectronApp(): boolean {
  return typeof window !== 'undefined' && Boolean(window.marineAPI);
}

/** Browser UI served from localhost (vite dev / preview), not Vercel. */
export function isLocalBrowserHost(): boolean {
  const h = browserHostname();
  return h === '' || h === 'localhost' || h === '127.0.0.1';
}

/** Hosted web UI (e.g. Vercel) — API must be set via VITE_API_URL at build time. */
export function isCloudBrowser(): boolean {
  if (typeof window === 'undefined') return false;
  if (isElectronApp()) return false;
  return !isLocalBrowserHost();
}

export type DeploymentTarget = 'local' | 'cloud' | 'electron';

export function deploymentTarget(): DeploymentTarget {
  if (isElectronApp()) return 'electron';
  if (isCloudBrowser()) return 'cloud';
  return 'local';
}

/** Vercel build without VITE_API_URL — UI will try localhost and fail. */
export function apiMisconfiguredForCloud(): boolean {
  return isCloudBrowser() && !import.meta.env.VITE_API_URL;
}

export function localFolderAvailable(): boolean {
  return isElectronApp();
}

export function backendOfflineHint(): string {
  if (apiMisconfiguredForCloud()) {
    return 'Cloud deploy: set VITE_API_URL to your Render API URL on Vercel, then redeploy.';
  }
  if (isCloudBrowser()) {
    return `Cannot reach API at ${apiBase()} — confirm Render is live and /health responds.`;
  }
  if (isElectronApp()) {
    return 'Backend offline — restart Annotra or run uvicorn on port 8765';
  }
  return 'Backend offline — run: npm run dev:backend (uvicorn on port 8765)';
}

export function wsBase(): string {
  return apiBase().replace(/^https:/, 'wss:').replace(/^http:/, 'ws:');
}

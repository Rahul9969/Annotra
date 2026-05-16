const STORAGE_KEY = 'marine_google_drive';

export interface DriveTokens {
  access_token: string;
  refresh_token?: string | null;
  expires_at?: number;
}

function load(): DriveTokens | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as DriveTokens;
  } catch {
    return null;
  }
}

function save(tokens: DriveTokens) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(tokens));
}

export function clearDriveTokens() {
  localStorage.removeItem(STORAGE_KEY);
}

export function isDriveProject(project: { source?: string } | null): boolean {
  return project?.source === 'drive';
}

async function refreshAccessToken(refreshToken: string): Promise<DriveTokens> {
  const { api } = await import('./api');
  const data = await api.driveOAuthRefresh(refreshToken);
  const prev = load();
  const merged: DriveTokens = {
    access_token: data.access_token,
    refresh_token: refreshToken,
    expires_at: data.expires_at,
  };
  if (prev?.refresh_token && !merged.refresh_token) {
    merged.refresh_token = prev.refresh_token;
  }
  save(merged);
  return merged;
}

/** Valid Google access token for Drive API calls (refreshes when needed). */
export async function getValidDriveAccessToken(): Promise<string | null> {
  let tokens = load();
  if (!tokens?.access_token) return null;
  const skew = 60;
  if (tokens.expires_at && tokens.expires_at > Date.now() / 1000 + skew) {
    return tokens.access_token;
  }
  if (tokens.refresh_token) {
    try {
      tokens = await refreshAccessToken(tokens.refresh_token);
      return tokens.access_token;
    } catch {
      clearDriveTokens();
      return null;
    }
  }
  return tokens.access_token;
}

export async function startDriveOAuth(): Promise<void> {
  const { api } = await import('./api');
  const { auth_url, state } = await api.driveOAuthStart();
  sessionStorage.setItem('marine_drive_oauth_state', state);
  window.open(auth_url, 'marine_drive_oauth', 'width=520,height=640');
}

/** Call on app load when URL contains drive_oauth_state= */
export async function completeDriveOAuthFromUrl(): Promise<boolean> {
  const params = new URLSearchParams(window.location.search);
  const state = params.get('drive_oauth_state');
  if (!state) return false;

  const expected = sessionStorage.getItem('marine_drive_oauth_state');
  if (expected && expected !== state) {
    console.warn('Drive OAuth state mismatch');
  }
  sessionStorage.removeItem('marine_drive_oauth_state');

  const { api } = await import('./api');
  const data = await api.driveOAuthResult(state);
  save({
    access_token: data.access_token,
    refresh_token: data.refresh_token ?? null,
    expires_at: data.expires_at,
  });

  params.delete('drive_oauth_state');
  const qs = params.toString();
  const next = `${window.location.pathname}${qs ? `?${qs}` : ''}${window.location.hash}`;
  window.history.replaceState({}, '', next);
  return true;
}

export function onDriveOAuthComplete(cb: () => void) {
  const handler = () => cb();
  window.addEventListener('marine-drive-oauth-done', handler);
  return () => window.removeEventListener('marine-drive-oauth-done', handler);
}

export function notifyDriveOAuthComplete() {
  window.dispatchEvent(new Event('marine-drive-oauth-done'));
}

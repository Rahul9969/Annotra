import {
  apiMisconfiguredForCloud,
  backendOfflineHint,
  isCloudBrowser,
} from './apiBase';
import type { ModelLoadState } from './hooks/useModelHealth';

export function modelsCanAnnotate(modelState: ModelLoadState): boolean {
  return modelState === 'ready';
}

export function modelStatusLabel(
  modelState: ModelLoadState,
  modelsLoaded: string[],
  modelError: string | null,
  backendOk: boolean,
): string {
  if (modelState === 'ready') return '';

  if (!backendOk || modelState === 'unknown') {
    if (apiMisconfiguredForCloud()) {
      return 'API URL not set — add VITE_API_URL on Vercel and redeploy';
    }
    if (isCloudBrowser()) {
      return 'API unreachable — check Render service';
    }
    return 'Backend offline — start API on port 8765';
  }

  if (modelState === 'loading' || modelState === 'not_loaded') {
    if (modelsLoaded.length > 0) {
      return `Loading models (${modelsLoaded.join(', ')})…`;
    }
    return isCloudBrowser()
      ? 'Loading AI models on server (first boot may take several minutes)…'
      : 'Loading AI models (best.pt, YOLO-World)…';
  }

  if (modelState === 'error') {
    const where = isCloudBrowser() ? 'Render logs' : 'backend terminal';
    return modelError ? `Model error: ${modelError}` : `Model load failed — check ${where}`;
  }

  return 'AI not ready';
}

export function modelStatusDetail(
  modelState: ModelLoadState,
  modelsLoaded: string[],
  modelError: string | null,
  backendOk: boolean,
): string {
  if (modelsCanAnnotate(modelState)) {
    return 'Auto-annotate current image (Ctrl+Shift+A)';
  }

  if (!backendOk || modelState === 'unknown') {
    return backendOfflineHint();
  }

  if (modelState === 'loading' || modelState === 'not_loaded') {
    const partial = modelsLoaded.length
      ? ` Loaded so far: ${modelsLoaded.join(', ')}.`
      : isCloudBrowser()
        ? ' First load on Render CPU can take 5–15 minutes.'
        : ' First load can take 1–3 minutes.';
    return `Auto is disabled while models load.${partial} Please wait.`;
  }

  if (modelState === 'error') {
    return `Auto is disabled: ${modelError ?? 'models failed to load'}. Fix backend/.env and restart.`;
  }

  return 'Auto is disabled until AI models are ready.';
}

import { useEffect, useState } from 'react';
import { api } from '../api';

export type ModelLoadState = 'unknown' | 'not_loaded' | 'loading' | 'ready' | 'error';

export function useModelHealth(pollMs = 1500) {
  const [modelState, setModelState] = useState<ModelLoadState>('unknown');
  const [modelsLoaded, setModelsLoaded] = useState<string[]>([]);
  const [modelError, setModelError] = useState<string | null>(null);
  const [backendOk, setBackendOk] = useState(false);

  useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      try {
        const h = await api.health();
        if (cancelled) return;
        setBackendOk(true);
        const status = (h.yolo_status ?? 'not_loaded') as ModelLoadState;
        setModelState(status);
        setModelsLoaded(h.models_loaded ?? []);
        setModelError(h.yolo_error ?? null);
        return status;
      } catch {
        if (!cancelled) {
          setBackendOk(false);
          setModelState('unknown');
          setModelsLoaded([]);
          setModelError(null);
        }
        return 'unknown';
      }
    };

    void poll();
    const id = window.setInterval(async () => {
      const status = await poll();
      if (status === 'ready' || status === 'error') {
        window.clearInterval(id);
      }
    }, pollMs);
    // Keep polling slowly after ready/error in case backend restarts
    const slowId = window.setInterval(() => void poll(), pollMs * 4);

    return () => {
      cancelled = true;
      window.clearInterval(id);
      window.clearInterval(slowId);
    };
  }, [pollMs]);

  const modelsReady = modelState === 'ready';

  return { modelState, modelsLoaded, modelError, modelsReady, backendOk };
}

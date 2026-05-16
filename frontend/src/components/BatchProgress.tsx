import { useEffect, useRef, useState } from 'react';
import { api } from '../api';
import { apiBase } from '../apiBase';
import type { BatchProgress as BP } from '../types';

export default function BatchProgressModal({
  jobId,
  onClose,
  onComplete,
}: {
  jobId: string;
  onClose: () => void;
  onComplete?: () => void | Promise<void>;
}) {
  const [progress, setProgress] = useState<BP | null>(null);
  const finishedRef = useRef(false);
  const onCompleteRef = useRef(onComplete);
  const onCloseRef = useRef(onClose);
  onCompleteRef.current = onComplete;
  onCloseRef.current = onClose;

  useEffect(() => {
    finishedRef.current = false;
    let cancelled = false;

    const finish = async (data: BP) => {
      if (finishedRef.current) return;
      if (data.status !== 'completed' && data.status !== 'cancelled') return;
      finishedRef.current = true;
      try {
        await onCompleteRef.current?.();
      } finally {
        setTimeout(() => onCloseRef.current(), 800);
      }
    };

    const poll = async () => {
      try {
        const data = await api.batchStatus(jobId);
        if (cancelled) return;
        setProgress(data);
        await finish(data);
      } catch {
        /* ignore transient errors */
      }
    };

    void poll();
    const interval = setInterval(() => void poll(), 800);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [jobId]);

  const pct = progress
    ? Math.min(100, Math.round((progress.completed / Math.max(progress.total, 1)) * 100))
    : 0;
  const done = progress?.status === 'completed' || progress?.status === 'cancelled';

  return (
    <div className="fixed bottom-4 right-4 z-50 glass rounded-xl p-4 w-80 shadow-2xl">
      <div className="flex justify-between items-center mb-2">
        <h3 className="text-sm font-semibold text-ocean-teal">Batch Annotation</h3>
        <button type="button" onClick={onClose} className="text-gray-400 hover:text-white">
          ×
        </button>
      </div>
      {progress && (
        <>
          <div className="h-2 bg-ocean-deep rounded overflow-hidden mb-2">
            <div className="h-full bg-ocean-teal transition-all" style={{ width: `${pct}%` }} />
          </div>
          <p className="text-xs text-gray-400">
            {done ? (
              <span className="text-ocean-teal">
                {progress.status === 'completed' ? 'Done' : 'Cancelled'} — {progress.completed}/{progress.total}
                {progress.failed > 0 ? ` (${progress.failed} failed)` : ''}
              </span>
            ) : (
              <>
                {progress.completed}/{progress.total} · {progress.images_per_sec} img/s
                {progress.completed > 0 ? ` · ETA ${Math.max(0, progress.eta_seconds)}s` : ''}
              </>
            )}
          </p>
          {!done && progress.completed === 0 && progress.status === 'running' && (
            <p className="text-xs text-amber-300/90 mt-1">
              Running on CPU — first image can take 1–3 minutes. Do not close the app.
            </p>
          )}
          <p className="text-xs truncate mt-1 text-gray-500">
            {progress.current_image ? `Processing: ${progress.current_image}` : 'Starting…'}
          </p>
          {!done && (
            <div className="flex gap-2 mt-3">
              <button
                type="button"
                className="text-xs px-2 py-1 rounded border border-ocean-border"
                onClick={() => fetch(`${apiBase()}/batch/${jobId}/pause`, { method: 'POST' })}
              >
                Pause
              </button>
              <button
                type="button"
                className="text-xs px-2 py-1 rounded border border-red-800 text-red-300"
                onClick={() => fetch(`${apiBase()}/batch/${jobId}/cancel`, { method: 'POST' })}
              >
                Cancel
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
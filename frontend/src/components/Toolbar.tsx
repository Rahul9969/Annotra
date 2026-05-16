import type { ModelLoadState } from '../hooks/useModelHealth';
import { modelStatusDetail, modelStatusLabel } from '../modelStatus';
import type { ToolMode } from '../types';
import { useStore } from '../store';
import AnnotraBrand from './AnnotraBrand';

const TOOLS: { id: ToolMode; label: string; key: string; title?: string }[] = [
  { id: 'box', label: '▭ Box', key: 'B' },
  { id: 'smart', label: '✦ Smart', key: 'S', title: 'SAM click segment (Shift+click background)' },
  { id: 'polygon', label: '⬡ Poly', key: 'P', title: 'Polygon — Enter or double-click to close' },
  { id: 'magic', label: '🪄 Magic', key: 'M', title: 'Magic wand flood-fill' },
  { id: 'select', label: '↕ Select', key: 'V' },
  { id: 'pan', label: '✋ Pan', key: ' ' },
];

interface ToolbarProps {
  onOpenFolder: () => void;
  onDashboard?: () => void;
  onShareProject?: () => void;
  isDriveProject?: boolean;
  onRefreshDrive?: () => void;
  onImportFromDrive?: () => void;
  onClearDriveCache?: () => void;
  onLinkLocalFolder?: () => void;
  localMirrorLinked?: boolean;
  onSave: () => void;
  onAutoAnnotate: () => void;
  onBatchAnnotate: () => void;
  onExport: () => void;
  onSettings: () => void;
  onStats: () => void;
  onPrev: () => void;
  onNext: () => void;
  modelsReady?: boolean;
  modelState?: ModelLoadState;
  modelsLoaded?: string[];
  modelError?: string | null;
  backendOk?: boolean;
}

export default function Toolbar({
  onOpenFolder,
  onDashboard,
  onShareProject,
  isDriveProject = false,
  onRefreshDrive,
  onImportFromDrive,
  onClearDriveCache,
  onLinkLocalFolder,
  localMirrorLinked = false,
  onSave,
  onAutoAnnotate,
  onBatchAnnotate,
  onExport,
  onSettings,
  onStats,
  onPrev,
  onNext,
  modelsReady = true,
  modelState = 'ready',
  modelsLoaded = [],
  modelError = null,
  backendOk = true,
}: ToolbarProps) {
  const { tool, setTool, zoom, setZoom, currentImageIndex, imagesTotal, undo, redo } = useStore();
  const aiDisabled = !modelsReady;
  const statusLabel = modelStatusLabel(modelState, modelsLoaded, modelError, backendOk);
  const autoTitle = modelStatusDetail(modelState, modelsLoaded, modelError, backendOk);

  return (
    <header className="h-12 border-b border-ocean-border bg-ocean-card flex items-center gap-2 px-3 shrink-0">
      <AnnotraBrand size="sm" />
      {onDashboard && (
        <button
          type="button"
          onClick={onDashboard}
          className="px-2 py-1 rounded text-xs hover:bg-ocean-deep text-ocean-teal ml-1"
          title="Back to projects"
        >
          ← Projects
        </button>
      )}
      {onShareProject && (
        <button
          type="button"
          onClick={onShareProject}
          className="px-2 py-1 rounded text-xs hover:bg-ocean-deep border border-ocean-border/80"
          title="Copy share code for collaborators"
        >
          🔗 Share
        </button>
      )}
      {!isDriveProject && (
        <button type="button" onClick={onOpenFolder} className="px-2 py-1 rounded text-xs hover:bg-ocean-deep" title="Open folder">
          📁 Open
        </button>
      )}
      {isDriveProject && onLinkLocalFolder && (
        <button
          type="button"
          onClick={onLinkLocalFolder}
          className={`px-2 py-1 rounded text-xs hover:bg-ocean-deep ${
            localMirrorLinked ? 'text-emerald-300 border border-emerald-700/50' : ''
          }`}
          title="Link the same dataset folder on this PC for faster batch/export"
        >
          {localMirrorLinked ? '📂 Local linked' : '📂 Link local'}
        </button>
      )}
      {isDriveProject && onRefreshDrive && (
        <button
          type="button"
          onClick={onRefreshDrive}
          className="px-2 py-1 rounded text-xs hover:bg-ocean-deep"
          title="Re-scan Google Drive folder for new images"
        >
          ☁ Refresh
        </button>
      )}
      {isDriveProject && onImportFromDrive && (
        <button
          type="button"
          onClick={onImportFromDrive}
          className="px-2 py-1 rounded text-xs hover:bg-ocean-deep"
          title="Load annotations from .marine-studio/annotations on Drive"
        >
          ⬇ Import
        </button>
      )}
      {isDriveProject && onClearDriveCache && (
        <button
          type="button"
          onClick={onClearDriveCache}
          className="px-2 py-1 rounded text-xs hover:bg-ocean-deep text-gray-400"
          title="Clear local image cache"
        >
          🗑 Cache
        </button>
      )}
      <button type="button" onClick={onSave} className="px-2 py-1 rounded text-xs hover:bg-ocean-deep transition">
        💾 Save
      </button>
      <button type="button" onClick={undo} className="px-2 py-1 rounded text-xs hover:bg-ocean-deep transition">
        ↩ Undo
      </button>
      <button type="button" onClick={redo} className="px-2 py-1 rounded text-xs hover:bg-ocean-deep transition">
        ↪ Redo
      </button>
      <span className="w-px h-6 bg-ocean-border" />
      <button
        type="button"
        onClick={() => {
          if (!aiDisabled) onAutoAnnotate();
        }}
        disabled={aiDisabled}
        aria-disabled={aiDisabled}
        title={autoTitle}
        className={`px-2 py-1 rounded text-xs transition text-ocean-teal ${
          aiDisabled ? 'opacity-40 cursor-not-allowed pointer-events-none' : 'hover:bg-ocean-deep'
        }`}
      >
        🤖 Auto
      </button>
      <button
        type="button"
        onClick={() => {
          if (!aiDisabled) onBatchAnnotate();
        }}
        disabled={aiDisabled}
        aria-disabled={aiDisabled}
        title={aiDisabled ? autoTitle.replace('Auto', 'Batch') : 'Batch annotate all images (Ctrl+Shift+B)'}
        className={`px-2 py-1 rounded text-xs transition ${
          aiDisabled ? 'opacity-40 cursor-not-allowed pointer-events-none' : 'hover:bg-ocean-deep'
        }`}
      >
        Batch
      </button>
      {aiDisabled && statusLabel && (
        <span
          className={`text-[10px] max-w-[220px] leading-tight ${
            modelState === 'error' ? 'text-amber-200' : 'text-amber-300/90 animate-pulse'
          }`}
          title={autoTitle}
        >
          {statusLabel}
        </span>
      )}
      <button type="button" onClick={onSettings} className="px-2 py-1 rounded text-xs hover:bg-ocean-deep transition">
        ⚙ AI
      </button>
      <button type="button" onClick={onStats} className="px-2 py-1 rounded text-xs hover:bg-ocean-deep transition">
        📊 Stats
      </button>
      <button type="button" onClick={onExport} className="px-2 py-1 rounded text-xs hover:bg-ocean-deep transition">
        📤 Export
      </button>

      <span className="w-px h-6 bg-ocean-border mx-1" />

      {TOOLS.map((t) => (
        <button
          key={t.id}
          type="button"
          onClick={() => setTool(t.id)}
          className={`px-2 py-1 rounded text-xs hover:bg-ocean-deep transition ${tool === t.id ? 'ring-1 ring-ocean-teal bg-ocean-deep' : ''}`}
          title={t.title ?? t.key}
        >
          {t.label}
        </button>
      ))}

      <span className="flex-1" />

      <button type="button" className="px-2 py-1 rounded text-xs hover:bg-ocean-deep transition" onClick={() => setZoom(zoom / 1.15)}>
        −
      </button>
      <span className="text-xs font-mono w-14 text-center text-ocean-teal">{Math.round(zoom * 100)}%</span>
      <button type="button" className="px-2 py-1 rounded text-xs hover:bg-ocean-deep transition" onClick={() => setZoom(zoom * 1.15)}>
        +
      </button>

      <span className="text-xs font-mono text-gray-400 mx-2">
        {currentImageIndex + 1}/{imagesTotal}
      </span>
      <button
        type="button"
        onClick={onPrev}
        title="Previous image (← or [)"
        className="px-2 py-1 rounded text-xs hover:bg-ocean-deep transition"
      >
        ◄ Prev
      </button>
      <button
        type="button"
        onClick={onNext}
        title="Next image (→, Enter, or ])"
        className="px-2 py-1 rounded text-xs hover:bg-ocean-deep transition"
      >
        Next ►
      </button>
    </header>
  );
}

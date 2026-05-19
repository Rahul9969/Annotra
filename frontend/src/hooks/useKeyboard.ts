import { useEffect } from 'react';
import { useStore } from '../store';
import type { ToolMode } from '../types';

const TOOL_KEYS: Record<string, ToolMode> = {
  b: 'box',
  s: 'smart',
  p: 'polygon',
  m: 'magic',
  v: 'select',
};

export function useKeyboard(handlers: {
  onPrev: () => void;
  onNext: () => void;
  onSave: () => void;
  onAutoAnnotate: () => void;
  onBatchAnnotate: () => void;
  onFit: () => void;
  onDelete: () => void;
  modelsReady?: boolean;
}) {
  const { setTool, undo, redo, selectAnnotation, annotations, selectedId, setAnnotations, showLabels, showBoxes } =
    useStore();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT') return;

      const mod = e.ctrlKey || e.metaKey;

      if (mod && e.key === 'z') {
        e.preventDefault();
        e.shiftKey ? redo() : undo();
        return;
      }
      if (mod && e.key === 'y') {
        e.preventDefault();
        redo();
        return;
      }
      if (mod && e.key === 's') {
        e.preventDefault();
        handlers.onSave();
        return;
      }
      if (mod && e.shiftKey && e.key === 'A') {
        e.preventDefault();
        if (handlers.modelsReady !== false) handlers.onAutoAnnotate();
        return;
      }
      if (mod && e.shiftKey && e.key === 'B') {
        e.preventDefault();
        if (handlers.modelsReady !== false) handlers.onBatchAnnotate();
        return;
      }

      const k = e.key.toLowerCase();
      if (TOOL_KEYS[k]) {
        setTool(TOOL_KEYS[k]);
        return;
      }

      if (e.key === 'ArrowLeft' || e.key === '[') {
        e.preventDefault();
        handlers.onPrev();
        return;
      }
      if (e.key === 'ArrowRight' || e.key === ']') {
        e.preventDefault();
        handlers.onNext();
        return;
      }
      if (e.key === 'Enter' && !e.shiftKey && !mod) {
        if ((e.target as HTMLElement).closest('[data-file-row]')) return;
        e.preventDefault();
        handlers.onNext();
        return;
      }
      if (e.key === 'Enter' && e.shiftKey) {
        e.preventDefault();
        handlers.onPrev();
        return;
      }
      if (e.key === 'f' || e.key === 'F') handlers.onFit();
      if (e.key === 'Delete') {
        handlers.onDelete();
        return;
      }
      if (e.key === 'PageDown' && selectedId != null) {
        e.preventDefault();
        handlers.onDelete();
        return;
      }
      if (e.key === 'h' || e.key === 'H') useStore.setState({ showBoxes: !showBoxes });
      if (e.key === 'l' || e.key === 'L') useStore.setState({ showLabels: !showLabels });
      if (e.key === 'Tab') {
        e.preventDefault();
        const ids = annotations.map((a, i) => a.id ?? i);
        if (!ids.length) return;
        const idx = selectedId != null ? ids.indexOf(selectedId) : -1;
        selectAnnotation(ids[(idx + 1) % ids.length]);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [handlers, annotations, selectedId, setTool, undo, redo, selectAnnotation, showBoxes, showLabels, setAnnotations]);
}

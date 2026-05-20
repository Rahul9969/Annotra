import { useMemo, useState } from 'react';
import { api } from '../api';
import { useStore } from '../store';

function isSegmentSource(source?: string): boolean {
  if (!source) return false;
  return /magic|sam|smart|polygon|segment|grabcut/i.test(source);
}

export default function RightPanel() {
  const {
    annotations,
    selectedId,
    classes,
    project,
    updateSelected,
    setAnnotations,
    setClasses,
  } = useStore();
  const [renameTo, setRenameTo] = useState('');
  const [renameBusy, setRenameBusy] = useState(false);

  const selected = annotations.find((a, i) => (a.id ?? i) === selectedId);
  const speciesInImage = new Set(annotations.map((a) => a.class_name));
  const multiSpecies = speciesInImage.size > 1;

  const classOptions = useMemo(() => {
    const names = new Set<string>();
    for (const c of classes) names.add(c.name);
    for (const a of annotations) {
      if (a.class_name) names.add(a.class_name);
    }
    if (!names.size) {
      names.add('unknown');
      names.add('fish');
    }
    return Array.from(names).sort((a, b) => a.localeCompare(b));
  }, [classes, annotations]);

  const removeSelected = () => {
    const { selectedId: sid, annotations: anns, setAnnotations: setAnns, selectAnnotation } =
      useStore.getState();
    if (sid == null) return;
    setAnns(anns.filter((a, i) => (a.id ?? i) !== sid), true);
    selectAnnotation(null);
  };

  const applyClassToSelected = (name: string) => {
    const trimmed = name.trim();
    if (!trimmed || !selected) return;
    updateSelected({ class_name: trimmed });
    if (!classOptions.includes(trimmed)) {
      setClasses([...classes, { id: 0, name: trimmed, color: '#00F5D4' }]);
    }
  };

  const renameClassProjectWide = async () => {
    if (!project || !selected) return;
    const next = renameTo.trim();
    const old = selected.class_name;
    if (!next || next === old) return;
    setRenameBusy(true);
    try {
      await api.renameClass(project.id, old, next);
      const refreshed = await api.listClasses(project.id);
      setClasses(refreshed);
      setAnnotations(
        annotations.map((a) => (a.class_name === old ? { ...a, class_name: next } : a)),
      );
      setRenameTo('');
    } catch (e) {
      alert(e instanceof Error ? e.message : 'Rename failed');
    } finally {
      setRenameBusy(false);
    }
  };

  return (
    <aside className="w-72 border-l border-ocean-border bg-ocean-card flex flex-col shrink-0">
      <div className="p-3 border-b border-ocean-border">
        <h2 className="text-sm font-semibold text-ocean-teal">
          ANNOTATIONS ({annotations.length})
        </h2>
      </div>

      {multiSpecies && (
        <p className="px-3 py-2 text-[10px] text-amber-200/90 border-b border-ocean-border leading-snug">
          Multiple species in this image — each box has its own class.
        </p>
      )}

      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {annotations.map((a, i) => {
          const id = a.id ?? i;
          const color = classes.find((c) => c.name === a.class_name)?.color ?? '#00F5D4';
          const active = selectedId === id;
          const hasMask = Boolean(a.polygon && a.polygon.length >= 3);
          return (
            <button
              key={id}
              type="button"
              onClick={() => useStore.getState().selectAnnotation(id)}
              className={`w-full text-left rounded-lg px-2 py-2 text-sm transition ${
                active ? 'glass ring-1 ring-ocean-teal' : 'hover:bg-ocean-deep'
              }`}
            >
              <span style={{ color }}>●</span> {a.class_name}{' '}
              <span className="text-gray-400">{(a.confidence * 100).toFixed(0)}%</span>
              {hasMask && (
                <span className="ml-1 text-[9px] text-ocean-teal/80 uppercase">mask</span>
              )}
            </button>
          );
        })}
      </div>

      {selected && (
        <div className="p-3 border-t border-ocean-border space-y-3 text-sm">
          <h3 className="font-semibold text-gray-300">SELECTED</h3>

          {selected.polygon && selected.polygon.length >= 3 && (
            <p className="text-[10px] text-ocean-teal/90 bg-ocean-deep/80 rounded px-2 py-1">
              Segmentation mask ({selected.polygon.length} points) — colored overlay on the fish
            </p>
          )}

          <label className="block">
            Class label
            <input
              list="annotra-class-list"
              className="w-full mt-1 bg-ocean-deep border border-ocean-border rounded px-2 py-1"
              value={selected.class_name}
              onChange={(e) => applyClassToSelected(e.target.value)}
              placeholder="Type or pick a species name"
            />
            <datalist id="annotra-class-list">
              {classOptions.map((n) => (
                <option key={n} value={n} />
              ))}
            </datalist>
          </label>

          <div className="rounded-lg border border-ocean-border/80 bg-ocean-deep/50 p-2 space-y-2">
            <div className="text-[10px] uppercase tracking-wide text-gray-500">Rename class (project)</div>
            <p className="text-[10px] text-gray-500 leading-snug">
              Renames &quot;{selected.class_name}&quot; on every image in this project.
            </p>
            <input
              className="w-full bg-ocean-deep border border-ocean-border rounded px-2 py-1 text-sm"
              placeholder="New class name"
              value={renameTo}
              onChange={(e) => setRenameTo(e.target.value)}
            />
            <button
              type="button"
              disabled={renameBusy || !renameTo.trim() || renameTo.trim() === selected.class_name}
              onClick={() => void renameClassProjectWide()}
              className="w-full py-1 rounded text-xs bg-ocean-teal/20 border border-ocean-teal/50 text-ocean-teal hover:bg-ocean-teal/30 disabled:opacity-40"
            >
              {renameBusy ? 'Renaming…' : 'Rename all boxes with this class'}
            </button>
          </div>

          <div className="font-mono text-xs space-y-1 text-gray-400">
            <div>Confidence: {selected.confidence.toFixed(2)}</div>
            <div>Source: {selected.source ?? 'human'}</div>
            <div>
              Box: {Math.round(selected.x)}, {Math.round(selected.y)} · {Math.round(selected.w)}×
              {Math.round(selected.h)}
            </div>
            {isSegmentSource(selected.source) && (
              <div className="text-ocean-teal/80">Magic / Smart replaces the last auto-mask</div>
            )}
          </div>

          <div className="flex gap-2">
            <button
              type="button"
              className="flex-1 py-1 rounded bg-ocean-deep border border-ocean-border text-xs"
              onClick={() => updateSelected({ locked: !selected.locked })}
            >
              {selected.locked ? 'Unlock' : 'Lock'}
            </button>
            <button
              type="button"
              className="flex-1 py-1 rounded bg-red-900/40 border border-red-800 text-xs"
              onClick={removeSelected}
            >
              Delete (Del / PgDn)
            </button>
          </div>

          <div className="pt-2 border-t border-ocean-border">
            <h3 className="font-semibold text-gray-300 mb-2">ATTRIBUTES</h3>
            {(['occluded', 'truncated', 'crowd'] as const).map((attr) => (
              <label key={attr} className="flex items-center gap-2 capitalize text-xs">
                <input
                  type="checkbox"
                  checked={Boolean(selected.attributes?.[attr])}
                  onChange={(e) =>
                    updateSelected({
                      attributes: { ...selected.attributes, [attr]: e.target.checked },
                    })
                  }
                />
                {attr}
              </label>
            ))}
          </div>
        </div>
      )}
    </aside>
  );
}

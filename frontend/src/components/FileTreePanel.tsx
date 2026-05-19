import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { VariableSizeList as List } from 'react-window';
import { useStore } from '../store';
import {
  buildSpeciesGroups,
  buildVisibleRows,
  speciesGroupKey,
  type TreeRow,
} from './fileTreeUtils';

const STATUS_ICON: Record<string, string> = {
  unannotated: '🔴',
  ai: '🟡',
  verified: '🟢',
  flagged: '⭐',
};

const ROW_H = 30;
const FOLDER_ROW_H = 34;

function rowHeight(row: TreeRow): number {
  return row.kind === 'folder' ? FOLDER_ROW_H : ROW_H;
}

export default function FileTreePanel({
  search,
  expanded,
  onExpandedChange,
  onSelect,
}: {
  search: string;
  expanded: Set<string>;
  onExpandedChange: (next: Set<string>) => void;
  onSelect: (index: number) => void;
}) {
  const { images, currentImageIndex, imagesTotal } = useStore();
  const containerRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<List>(null);
  const [listHeight, setListHeight] = useState(320);

  const groups = useMemo(() => buildSpeciesGroups(images, search), [images, search]);
  const rows = useMemo(() => buildVisibleRows(groups, expanded), [groups, expanded]);

  const totalRowsHeight = useMemo(() => rows.reduce((sum, row) => sum + rowHeight(row), 0), [rows]);
  const viewportHeight = rows.length ? Math.min(listHeight, Math.max(ROW_H, totalRowsHeight)) : 0;

  const getItemSize = useCallback((index: number) => rowHeight(rows[index]), [rows]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setListHeight(Math.max(120, el.clientHeight)));
    ro.observe(el);
    setListHeight(Math.max(120, el.clientHeight));
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    listRef.current?.resetAfterIndex(0);
  }, [rows]);

  useEffect(() => {
    if (!search.trim()) return;
    onExpandedChange(new Set(groups.map((g) => g.key)));
  }, [search, groups, onExpandedChange]);

  useEffect(() => {
    const idx = rows.findIndex((r) => r.kind === 'file' && r.imageIndex === currentImageIndex);
    if (idx >= 0) listRef.current?.scrollToItem(idx, 'smart');
  }, [currentImageIndex, rows]);

  const toggleFolder = useCallback(
    (key: string) => {
      const next = new Set(expanded);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      onExpandedChange(next);
    },
    [expanded, onExpandedChange],
  );

  const expandAll = () => onExpandedChange(new Set(groups.map((g) => g.key)));
  const collapseAll = () => onExpandedChange(new Set());

  if (!images.length) {
    return <p className="p-3 text-xs text-gray-500">No images — open a folder</p>;
  }

  if (!groups.length) {
    return <p className="p-3 text-xs text-gray-500">No files match your search</p>;
  }

  const Row = ({ index, style }: { index: number; style: React.CSSProperties }) => {
    const row = rows[index];
    if (row.kind === 'folder') {
      return (
        <div style={style} className="px-1">
          <button
            type="button"
            onClick={() => toggleFolder(row.key)}
            className="w-full flex items-center gap-1.5 px-2 py-1.5 rounded text-left text-xs bg-ocean-deep/80 hover:bg-ocean-deep border border-ocean-border/60"
            title={row.expanded ? 'Collapse folder' : 'Expand folder'}
          >
            <span className="text-gray-400 w-3 shrink-0 font-mono">{row.expanded ? '▼' : '▶'}</span>
            <span className="text-ocean-teal font-semibold truncate flex-1" title={row.label}>
              {row.label}
            </span>
            <span className="text-[10px] text-gray-500 shrink-0 font-mono">
              {row.annotated}/{row.total}
            </span>
          </button>
        </div>
      );
    }

    const active = row.imageIndex === currentImageIndex;
    return (
      <div style={style} className="px-1">
        <div
          data-file-row
          role="button"
          tabIndex={0}
          onClick={() => onSelect(row.imageIndex)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.stopPropagation();
              onSelect(row.imageIndex);
            }
          }}
          className={`flex items-center gap-2 pl-7 pr-2 py-1 text-xs cursor-pointer rounded border-l-2 ${
            active ? 'bg-ocean-blue/25 border-ocean-teal' : 'border-transparent hover:bg-ocean-card/80'
          }`}
          title={row.label}
        >
          <span className="shrink-0 text-[10px]">{STATUS_ICON[row.status] ?? '🔴'}</span>
          <span className="truncate flex-1 text-gray-200">{row.label}</span>
          {row.annotation_count > 0 && (
            <span className="text-ocean-teal font-mono text-[10px] shrink-0">{row.annotation_count}</span>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex items-center gap-1 px-2 py-1.5 border-b border-ocean-border/50 shrink-0">
        <span className="text-[10px] text-gray-500 flex-1" title={`${images.length} of ${imagesTotal} images in sidebar`}>
          {groups.length} folder{groups.length === 1 ? '' : 's'}
          {imagesTotal > 0 ? ` · ${imagesTotal} imgs` : ''}
        </span>
        <button
          type="button"
          onClick={expandAll}
          className="px-1.5 py-0.5 rounded text-[10px] text-gray-400 hover:text-ocean-teal hover:bg-ocean-deep"
        >
          Expand all
        </button>
        <button
          type="button"
          onClick={collapseAll}
          className="px-1.5 py-0.5 rounded text-[10px] text-gray-400 hover:text-ocean-teal hover:bg-ocean-deep"
        >
          Collapse all
        </button>
      </div>

      <div ref={containerRef} className="flex-1 min-h-0 flex flex-col overflow-auto">
        <List
          ref={listRef}
          height={viewportHeight || 1}
          width="100%"
          itemCount={rows.length}
          itemSize={getItemSize}
          overscanCount={12}
        >
          {Row}
        </List>
      </div>
    </div>
  );
}

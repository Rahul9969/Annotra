import type { ImageItem, ImageStatus } from '../types';

const GENERIC_GROUP = new Set(['unknown', 'uncategorized', 'unclassified', 'misc', 'other']);

export function speciesGroupKey(im: ImageItem): string {
  if (im.species_class?.trim()) return im.species_class.trim();
  const rel = im.rel_path?.replace(/\\/g, '/');
  if (rel) {
    const parts = rel.split('/').filter(Boolean);
    if (parts.length > 1) {
      const parent = parts[parts.length - 2];
      if (parent && !['train', 'val', 'test', 'images'].includes(parent.toLowerCase())) {
        return parent;
      }
    }
    const fname = parts[parts.length - 1] ?? '';
    const m = fname.match(/^([a-z][a-z0-9]*_[a-z][a-z0-9]*)/i);
    if (m) return m[1];
  }
  return 'Uncategorized';
}

/** Default class for new boxes — prefer current image folder, not classes[0]. */
export function resolveDefaultClassName(
  im: ImageItem | null,
  classNames: { name: string }[],
  existingAnnClass?: string,
): string {
  const pick = (name: string) => {
    const n = name.trim();
    if (!n || GENERIC_GROUP.has(n.toLowerCase())) return null;
    if (classNames.some((c) => c.name === n)) return n;
    return n;
  };
  if (im) {
    const fromSpecies = pick(im.species_class ?? '');
    if (fromSpecies) return fromSpecies;
    const fromGroup = pick(speciesGroupKey(im));
    if (fromGroup) return fromGroup;
  }
  if (existingAnnClass) {
    const fromAnn = pick(existingAnnClass);
    if (fromAnn) return fromAnn;
  }
  return classNames[0]?.name ?? 'fish';
}

export function imageFileName(im: ImageItem): string {
  const rel = (im.rel_path ?? im.path).replace(/\\/g, '/');
  const parts = rel.split('/').filter(Boolean);
  return parts[parts.length - 1] ?? rel;
}

export interface SpeciesGroup {
  key: string;
  items: { image: ImageItem; index: number }[];
}

export type TreeRow =
  | {
      kind: 'folder';
      key: string;
      label: string;
      total: number;
      annotated: number;
      expanded: boolean;
    }
  | {
      kind: 'file';
      key: string;
      imageIndex: number;
      label: string;
      status: ImageStatus;
      annotation_count: number;
    };

export function buildSpeciesGroups(images: ImageItem[], search: string): SpeciesGroup[] {
  const q = search.trim().toLowerCase();
  const map = new Map<string, { image: ImageItem; index: number }[]>();

  images.forEach((image, index) => {
    if (q) {
      const hay = `${image.path} ${image.rel_path ?? ''} ${image.species_class ?? ''}`.toLowerCase();
      if (!hay.includes(q)) return;
    }
    const key = speciesGroupKey(image);
    if (!map.has(key)) map.set(key, []);
    map.get(key)!.push({ image, index });
  });

  return [...map.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, items]) => ({
      key,
      items: items.sort((x, y) => imageFileName(x.image).localeCompare(imageFileName(y.image))),
    }));
}

/** Flat image indices in sidebar order (folder sort → filename sort within folder). */
export function buildFlatNavigationOrder(
  images: ImageItem[],
  search: string,
  expanded: Set<string>,
): number[] {
  const groups = buildSpeciesGroups(images, search);
  const rows = buildVisibleRows(groups, expanded);
  return rows
    .filter((r): r is Extract<TreeRow, { kind: 'file' }> => r.kind === 'file')
    .map((r) => r.imageIndex);
}

export function buildVisibleRows(groups: SpeciesGroup[], expanded: Set<string>): TreeRow[] {
  const rows: TreeRow[] = [];
  for (const g of groups) {
    const annotated = g.items.filter(
      ({ image }) => image.status === 'ai' || image.status === 'verified',
    ).length;
    const isExpanded = expanded.has(g.key);
    rows.push({
      kind: 'folder',
      key: g.key,
      label: g.key,
      total: g.items.length,
      annotated,
      expanded: isExpanded,
    });
    if (isExpanded) {
      for (const { image, index } of g.items) {
        rows.push({
          kind: 'file',
          key: `${g.key}:${image.id}:${index}`,
          imageIndex: index,
          label: imageFileName(image),
          status: image.status,
          annotation_count: image.annotation_count,
        });
      }
    }
  }
  return rows;
}

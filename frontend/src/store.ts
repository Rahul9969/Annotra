import { create } from 'zustand';
import type { AISettings, BBox, ClassItem, ImageItem, ProjectInfo, ToolMode } from './types';

interface HistoryState {
  past: BBox[][];
  future: BBox[][];
}

interface AppState {
  project: ProjectInfo | null;
  images: ImageItem[];
  imagesTotal: number;
  currentImageIndex: number;
  currentImage: ImageItem | null;
  imageSrc: string | null;
  annotations: BBox[];
  selectedId: number | null;
  tool: ToolMode;
  zoom: number;
  pan: { x: number; y: number };
  classes: ClassItem[];
  aiSettings: AISettings;
  showLabels: boolean;
  showBoxes: boolean;
  boxOpacity: number;
  gridSnap: number;
  prompts: string;
  batchJobId: string | null;
  history: HistoryState;
  panel: 'annotate' | 'settings' | 'export' | 'train' | 'stats' | null;
  collaborationLocalRoot: string | null;

  setProject: (p: ProjectInfo | null) => void;
  setImages: (items: ImageItem[], total: number) => void;
  setCurrentIndex: (i: number) => void;
  setImageSrc: (src: string | null) => void;
  setAnnotations: (a: BBox[], pushHistory?: boolean) => void;
  selectAnnotation: (id: number | null) => void;
  setTool: (t: ToolMode) => void;
  setZoom: (z: number) => void;
  setPan: (p: { x: number; y: number }) => void;
  setClasses: (c: ClassItem[]) => void;
  setAISettings: (s: AISettings) => void;
  undo: () => void;
  redo: () => void;
  setPanel: (p: AppState['panel']) => void;
  setBatchJobId: (id: string | null) => void;
  setCollaborationLocalRoot: (root: string | null) => void;
  updateSelected: (patch: Partial<BBox>) => void;
}

export const useStore = create<AppState>((set, get) => ({
  project: null,
  images: [],
  imagesTotal: 0,
  currentImageIndex: 0,
  currentImage: null,
  imageSrc: null,
  annotations: [],
  selectedId: null,
  tool: 'select',
  zoom: 1,
  pan: { x: 0, y: 0 },
  classes: [],
  aiSettings: {
    confidence: 0.35,
    iou_threshold: 0.45,
    max_boxes: 300,
    batch_size: 8,
    thread_pool_size: 4,
    device: 'auto',
    half_precision: true,
    yolo_model: 'yolov8n.pt',
    annotation_mode: 'both',
  },
  showLabels: true,
  showBoxes: true,
  boxOpacity: 0.7,
  gridSnap: 8,
  prompts: 'fish, shark, coral, jellyfish, ray, turtle',
  batchJobId: null,
  history: { past: [], future: [] },
  panel: null,
  collaborationLocalRoot: null,

  setProject: (project) => set({ project }),
  setImages: (images, imagesTotal) => set({ images, imagesTotal }),
  setCurrentIndex: (currentImageIndex) => {
    const images = get().images;
    set({
      currentImageIndex,
      currentImage: images[currentImageIndex] ?? null,
      selectedId: null,
    });
  },
  setImageSrc: (imageSrc) => set({ imageSrc }),
  setAnnotations: (annotations, pushHistory = true) => {
    const { annotations: prev, history } = get();
    if (pushHistory) {
      set({
        annotations,
        history: {
          past: [...history.past, prev],
          future: [],
        },
      });
    } else {
      set({ annotations });
    }
  },
  selectAnnotation: (selectedId) => set({ selectedId }),
  setTool: (tool) => set({ tool }),
  setZoom: (zoom) => set({ zoom: Math.min(32, Math.max(0.1, zoom)) }),
  setPan: (pan) => set({ pan }),
  setClasses: (classes) => set({ classes }),
  setAISettings: (aiSettings) => set({ aiSettings }),
  undo: () => {
    const { history, annotations } = get();
    if (!history.past.length) return;
    const prev = history.past[history.past.length - 1];
    set({
      annotations: prev,
      history: {
        past: history.past.slice(0, -1),
        future: [annotations, ...history.future],
      },
    });
  },
  redo: () => {
    const { history, annotations } = get();
    if (!history.future.length) return;
    const next = history.future[0];
    set({
      annotations: next,
      history: {
        past: [...history.past, annotations],
        future: history.future.slice(1),
      },
    });
  },
  setPanel: (panel) => set({ panel }),
  setBatchJobId: (batchJobId) => set({ batchJobId }),
  setCollaborationLocalRoot: (collaborationLocalRoot) => set({ collaborationLocalRoot }),
  updateSelected: (patch) => {
    const { annotations, selectedId } = get();
    if (selectedId == null) return;
    set({
      annotations: annotations.map((a, i) =>
        (a.id ?? i) === selectedId ? { ...a, ...patch } : a,
      ),
    });
  },
}));

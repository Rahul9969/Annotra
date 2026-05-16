export type ToolMode = 'select' | 'box' | 'polygon' | 'smart' | 'magic' | 'pan';
export type ImageStatus = 'unannotated' | 'ai' | 'verified' | 'flagged';

export interface BBox {
  id?: number;
  class_id?: number;
  class_name: string;
  confidence: number;
  x: number;
  y: number;
  w: number;
  h: number;
  rotation: number;
  polygon?: number[][];
  attributes?: Record<string, unknown>;
  source?: string;
  z_index?: number;
  locked?: boolean;
  hidden?: boolean;
}

export interface ImageItem {
  id: number;
  path: string;
  rel_path?: string;
  drive_file_id?: string;
  species_class?: string;
  width: number;
  height: number;
  status: ImageStatus;
  annotation_count: number;
}

export interface ClassItem {
  id: number;
  name: string;
  color: string;
  hotkey?: string;
  supercategory?: string;
}

export interface ProjectInfo {
  id: number;
  name: string;
  root_path: string;
  image_count: number;
  annotated_count: number;
  source?: 'local' | 'drive';
  drive_folder_id?: string | null;
  local_mirror_path?: string | null;
}

export interface AISettings {
  confidence: number;
  iou_threshold: number;
  max_boxes: number;
  batch_size: number;
  thread_pool_size: number;
  device: string;
  half_precision: boolean;
  yolo_model: string;
  enable_dino?: boolean;
  enable_sam?: boolean;
  enable_clip?: boolean;
}

export interface BatchProgress {
  job_id: string;
  status: string;
  total: number;
  completed: number;
  failed: number;
  images_per_sec: number;
  eta_seconds: number;
  current_image?: string;
}

declare global {
  interface Window {
    marineAPI?: {
      openFolder: () => Promise<string | null>;
      scanFolder: (path: string, recursive?: boolean) => Promise<{ path: string; name: string; folder: string }[]>;
      readFileBase64: (path: string) => Promise<string>;
      joinDatasetPath?: (root: string, relPath: string) => Promise<string | null>;
      saveFolder: () => Promise<string | null>;
      exists: (path: string) => Promise<boolean>;
      showItemInFolder: (path: string) => Promise<void>;
      getPaths: () => Promise<{ backendUrl: string; userData: string }>;
    };
  }
}

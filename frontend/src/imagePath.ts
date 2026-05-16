import { api } from './api';
import { getValidDriveAccessToken } from './driveAuth';
import type { ImageItem } from './types';

/** Strip data URL prefix from Electron readFileBase64 result */
export function rawBase64FromDataUrl(dataUrl: string): string {
  const i = dataUrl.indexOf(',');
  return i >= 0 ? dataUrl.slice(i + 1) : dataUrl;
}

/** Absolute path to image file on this machine (owner uses DB path; collaborator uses local dataset root + rel_path). */
export async function resolveImageDiskPath(
  im: ImageItem,
  collaborationLocalRoot: string | null,
): Promise<string> {
  if (collaborationLocalRoot && window.marineAPI?.joinDatasetPath) {
    const rel = im.rel_path?.trim();
    if (rel) {
      const joined = await window.marineAPI.joinDatasetPath(collaborationLocalRoot, rel);
      if (joined) return joined;
    }
  }
  return im.path;
}

/** Load image as data URL for the canvas (local disk, collaborator mirror, or Google Drive stream). */
export async function resolveImageDisplaySrc(
  im: ImageItem,
  project: { source?: string } | null,
  collaborationLocalRoot: string | null,
): Promise<string> {
  if (collaborationLocalRoot && window.marineAPI) {
    try {
      const diskPath = await resolveImageDiskPath(im, collaborationLocalRoot);
      if (await window.marineAPI.exists(diskPath)) {
        return window.marineAPI.readFileBase64(diskPath);
      }
    } catch {
      /* use Drive or server path */
    }
  }

  if (project?.source === 'drive' && im.drive_file_id) {
    const token = await getValidDriveAccessToken();
    if (!token) {
      throw new Error('Connect Google Drive to view images from the cloud dataset.');
    }
    const res = await fetch(api.imageMediaUrl(im.id), {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) {
      const t = await res.text();
      throw new Error(t || 'Failed to load image from Google Drive');
    }
    const blob = await res.blob();
    return await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = () => reject(new Error('Failed to read image'));
      reader.readAsDataURL(blob);
    });
  }

  const diskPath = await resolveImageDiskPath(im, collaborationLocalRoot);
  if (window.marineAPI) {
    return window.marineAPI.readFileBase64(diskPath);
  }
  return `file://${diskPath}`;
}

import { rawBase64FromDataUrl, resolveImageDiskPath } from './imagePath';
import { isDriveProject } from './driveAuth';
import type { ImageItem, ProjectInfo } from './types';

export async function buildSegmentImagePayload(
  im: ImageItem,
  project: ProjectInfo | null,
  collaborationLocalRoot: string | null,
): Promise<{
  image_path: string;
  image_id: number;
  project_id?: number;
  image_base64?: string;
}> {
  let imageBase64: string | undefined;
  if (collaborationLocalRoot && window.marineAPI && project && !isDriveProject(project)) {
    const diskPath = await resolveImageDiskPath(im, collaborationLocalRoot);
    const dataUrl = await window.marineAPI.readFileBase64(diskPath);
    imageBase64 = rawBase64FromDataUrl(dataUrl);
  }
  return {
    image_path: im.path,
    image_id: im.id,
    project_id: project?.id,
    ...(imageBase64 ? { image_base64: imageBase64 } : {}),
  };
}

export function bboxFromPolygon(points: number[][]): { x: number; y: number; w: number; h: number } {
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const x = Math.min(...xs);
  const y = Math.min(...ys);
  return { x, y, w: Math.max(...xs) - x, h: Math.max(...ys) - y };
}

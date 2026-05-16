import { getValidDriveAccessToken } from './driveAuth';
import { apiBase, wsBase } from './apiBase';

type RequestOpts = RequestInit & { skipDriveAuth?: boolean };

async function request<T>(path: string, options?: RequestOpts): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options?.headers as Record<string, string> | undefined),
  };
  if (!options?.skipDriveAuth) {
    const token = await getValidDriveAccessToken();
    if (token) headers.Authorization = `Bearer ${token}`;
  }

  const res = await fetch(`${apiBase()}${path}`, {
    ...options,
    headers,
  });
  if (!res.ok) {
    const text = await res.text();
    let message = text || res.statusText;
    try {
      const parsed = JSON.parse(text) as { detail?: string | { msg?: string }[] };
      if (typeof parsed.detail === 'string') message = parsed.detail;
      else if (Array.isArray(parsed.detail)) {
        message = parsed.detail.map((d) => (typeof d === 'string' ? d : d.msg ?? '')).filter(Boolean).join('; ');
      }
    } catch {
      /* plain text error */
    }
    throw new Error(message);
  }
  return res.json();
}

export const api = {
  health: () =>
    request<{
      status: string;
      device: string;
      yolo_status?: string;
      yolo_error?: string | null;
      models_loaded?: string[];
      using_custom_weights?: boolean;
      weights_custom?: string;
      google_drive_oauth_configured?: boolean;
    }>('/health', { skipDriveAuth: true }),

  driveOAuthStart: () =>
    request<{ auth_url: string; state: string }>('/drive/oauth/start', { skipDriveAuth: true }),

  driveOAuthResult: (state: string) =>
    request<{ access_token: string; refresh_token?: string | null; expires_at?: number }>(
      `/drive/oauth/result?state=${encodeURIComponent(state)}`,
      { skipDriveAuth: true },
    ),

  driveOAuthRefresh: (refresh_token: string) =>
    request<{ access_token: string; expires_at?: number }>('/drive/oauth/refresh', {
      method: 'POST',
      body: JSON.stringify({ refresh_token }),
      skipDriveAuth: true,
    }),

  createProjectFromDrive: (name: string, folder_url: string, access_token: string) =>
    request<import('./types').ProjectInfo>('/projects/from-drive', {
      method: 'POST',
      body: JSON.stringify({ name, folder_url, access_token }),
      headers: { Authorization: `Bearer ${access_token}` },
    }),

  reindexDriveProject: (projectId: number) =>
    request<import('./types').ProjectInfo>(`/projects/${projectId}/drive/reindex`, { method: 'POST' }),

  driveImportAnnotations: (projectId: number, overwrite = false) =>
    request<{ imported: number; skipped: number; files: number }>(
      `/projects/${projectId}/drive/import-annotations?overwrite=${overwrite}`,
      { method: 'POST' },
    ),

  driveClearCache: (projectId: number) =>
    request<{ ok: boolean; bytes_freed: number }>(`/projects/${projectId}/drive/cache/clear`, {
      method: 'POST',
    }),

  driveCacheStats: (projectId: number) =>
    request<{ files: number; bytes: number }>(`/projects/${projectId}/drive/cache/stats`),

  setLocalMirror: (projectId: number, path: string | null) =>
    request<{ local_mirror_path: string | null }>(`/projects/${projectId}/local-mirror`, {
      method: 'PUT',
      body: JSON.stringify({ path }),
    }),

  imageMediaUrl: (imageId: number) => `${apiBase()}/images/${imageId}/media`,

  listProjects: () => request<import('./types').ProjectInfo[]>('/projects'),

  shareProject: (projectId: number) =>
    request<{ share_token: string }>(`/projects/${projectId}/share`, { method: 'POST' }),

  projectByShareToken: (token: string) =>
    request<{ project: import('./types').ProjectInfo; share_token: string }>(
      `/projects/by-share/${encodeURIComponent(token)}`,
    ),

  createProject: (name: string, root_path: string, class_names?: string[]) =>
    request('/projects', { method: 'POST', body: JSON.stringify({ name, root_path, class_names }) }),

  indexImages: (projectId: number, files: { path: string }[]) =>
    request(`/projects/${projectId}/index`, { method: 'POST', body: JSON.stringify(files) }),

  listImages: (projectId: number, offset = 0, limit = 100, search = '') =>
    request<{ total: number; items: import('./types').ImageItem[] }>(
      `/projects/${projectId}/images?offset=${offset}&limit=${limit}&search=${encodeURIComponent(search)}`,
    ),

  listAllImages: async (projectId: number, pageSize = 2000) => {
    const items: import('./types').ImageItem[] = [];
    let offset = 0;
    let total = 0;
    do {
      const page = await api.listImages(projectId, offset, pageSize);
      items.push(...page.items);
      total = page.total;
      offset += pageSize;
    } while (items.length < total);
    return { total, items };
  },

  getProject: (projectId: number) => request<import('./types').ProjectInfo>(`/projects/${projectId}`),

  deleteProject: (projectId: number) =>
    request<{ ok: boolean; deleted_project_id: number }>(`/projects/${projectId}`, {
      method: 'DELETE',
    }),

  getAnnotations: (imageId: number) => request<import('./types').BBox[]>(`/images/${imageId}/annotations`),

  saveAnnotations: (imageId: number, annotations: import('./types').BBox[], status?: string) =>
    request('/annotations/save', {
      method: 'POST',
      body: JSON.stringify({ image_id: imageId, annotations, status }),
    }),

  segmentMagic: (
    body: {
      image_path: string;
      image_id?: number;
      project_id?: number;
      image_base64?: string;
      x: number;
      y: number;
      tolerance?: number;
    },
  ) =>
    request<{
      polygon: number[][];
      x: number;
      y: number;
      w: number;
      h: number;
      width: number;
      height: number;
      source: string;
    }>('/segment/magic', { method: 'POST', body: JSON.stringify(body) }),

  segmentSmart: (
    body: {
      image_path: string;
      image_id?: number;
      project_id?: number;
      image_base64?: string;
      points: number[][];
      labels: number[];
    },
  ) =>
    request<{
      polygon: number[][];
      x: number;
      y: number;
      w: number;
      h: number;
      width: number;
      height: number;
      source: string;
    }>('/segment/smart', { method: 'POST', body: JSON.stringify(body) }),

  annotate: (
    image_path: string,
    image_id?: number,
    project_id?: number,
    prompts?: string[],
    image_base64?: string | null,
  ) =>
    request<{ annotations: import('./types').BBox[]; width: number; height: number; timing_ms: Record<string, number> }>(
      '/annotate',
      {
        method: 'POST',
        body: JSON.stringify({
          image_path,
          image_id,
          project_id,
          prompts,
          image_base64: image_base64 ?? undefined,
        }),
      },
    ),

  batchStart: (project_id: number, image_ids?: number[], skip_annotated = true, force_all = false) =>
    request<{ job_id: string; total: number }>('/batch/start', {
      method: 'POST',
      body: JSON.stringify({ project_id, image_ids, skip_annotated: force_all ? false : skip_annotated }),
    }),

  reindexProject: (projectId: number, files: { path: string; folder?: string }[]) =>
    request<{ added: number; classes_from_folders: string[] }>(`/projects/${projectId}/index`, {
      method: 'POST',
      body: JSON.stringify(files),
    }),

  reindexSpecies: (projectId: number) =>
    request<{ updated: number; classes: string[] }>(`/projects/${projectId}/reindex-species`, {
      method: 'POST',
    }),

  batchStatus: (jobId: string) => request<import('./types').BatchProgress>(`/batch/${jobId}/status`),

  listClasses: (projectId: number) => request<import('./types').ClassItem[]>(`/projects/${projectId}/classes`),

  renameClass: (projectId: number, old_name: string, new_name: string) =>
    request<{ ok: boolean; old_name: string; new_name: string; annotations_updated: number }>(
      `/projects/${projectId}/classes/rename`,
      { method: 'PATCH', body: JSON.stringify({ old_name, new_name }) },
    ),

  stats: (projectId: number) => request<Record<string, unknown>>(`/projects/${projectId}/stats`),

  export: (body: Record<string, unknown>) =>
    request<{
      output_dir: string;
      zip_path?: string | null;
      manifest: string;
      counts: Record<string, number>;
      source_stats?: Record<string, number>;
      total_exported?: number;
    }>('/export', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  getAISettings: () => request<import('./types').AISettings>('/ai/settings'),

  updateAISettings: (settings: import('./types').AISettings) =>
    request('/ai/settings', { method: 'PUT', body: JSON.stringify(settings) }),

  batchWsUrl: (jobId: string) => `${wsBase()}/ws/batch/${jobId}`,

  projectWsUrl: (projectId: number) => `${wsBase()}/ws/project/${projectId}`,
};

import { useCallback, useEffect, useState } from 'react';
import { api } from '../api';
import AnnotraBrand from './AnnotraBrand';
import {
  getValidDriveAccessToken,
  isDriveProject,
  onDriveOAuthComplete,
  startDriveOAuth,
} from '../driveAuth';
import type { ProjectInfo } from '../types';

export default function Dashboard({
  onOpenProject,
  onCreateNew,
  onCreateDrive,
}: {
  onOpenProject: (projectId: number, collaborationLocalRoot: string | null) => Promise<void>;
  onCreateNew: () => void;
  onCreateDrive: (projectId: number) => Promise<void>;
}) {
  const [projects, setProjects] = useState<ProjectInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [joinOpen, setJoinOpen] = useState(false);
  const [joinToken, setJoinToken] = useState('');
  const [joinPreview, setJoinPreview] = useState<ProjectInfo | null>(null);
  const [joinBusy, setJoinBusy] = useState(false);
  const [driveOpen, setDriveOpen] = useState(false);
  const [driveName, setDriveName] = useState('');
  const [driveUrl, setDriveUrl] = useState('');
  const [driveBusy, setDriveBusy] = useState(false);
  const [driveConfigured, setDriveConfigured] = useState(false);
  const [driveConnected, setDriveConnected] = useState(false);

  const refreshDriveStatus = useCallback(async () => {
    try {
      const h = await api.health();
      setDriveConfigured(Boolean(h.google_drive_oauth_configured));
    } catch {
      setDriveConfigured(false);
    }
    setDriveConnected(Boolean(await getValidDriveAccessToken()));
  }, []);

  const refresh = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      setProjects(await api.listProjects());
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'Failed to load projects');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    void refreshDriveStatus();
  }, [refresh, refreshDriveStatus]);

  useEffect(() => onDriveOAuthComplete(() => void refreshDriveStatus()), [refreshDriveStatus]);

  const lookupShare = async () => {
    const t = joinToken.trim();
    if (!t) return;
    setJoinBusy(true);
    setErr(null);
    try {
      const res = await api.projectByShareToken(t);
      setJoinPreview(res.project);
    } catch (e) {
      setJoinPreview(null);
      setErr(e instanceof Error ? e.message : 'Invalid share code');
    } finally {
      setJoinBusy(false);
    }
  };

  const finishJoin = async () => {
    if (!joinPreview) return;
    if (isDriveProject(joinPreview)) {
      const token = await getValidDriveAccessToken();
      if (!token) {
        alert('Connect Google Drive first (same account that can open the shared folder).');
        return;
      }
      setJoinBusy(true);
      try {
        await onOpenProject(joinPreview.id, null);
        setJoinOpen(false);
        setJoinToken('');
        setJoinPreview(null);
      } finally {
        setJoinBusy(false);
      }
      return;
    }
    if (!window.marineAPI) {
      alert('Desktop app required to pick your local copy of the dataset.');
      return;
    }
    const folder = await window.marineAPI.openFolder();
    if (!folder) return;
    setJoinBusy(true);
    try {
      await api.setLocalMirror(joinPreview.id, folder);
      await onOpenProject(joinPreview.id, folder);
      setJoinOpen(false);
      setJoinToken('');
      setJoinPreview(null);
    } finally {
      setJoinBusy(false);
    }
  };

  const deleteProject = async (p: ProjectInfo, e: React.MouseEvent) => {
    e.stopPropagation();
    const label = isDriveProject(p) ? 'cloud' : 'local';
    if (
      !window.confirm(
        `Delete project "${p.name}"?\n\nThis removes it from Annotra (${label} dataset files on disk/Drive are not deleted). Annotations in the database for this project will be lost.`,
      )
    ) {
      return;
    }
    setErr(null);
    try {
      await api.deleteProject(p.id);
      await refresh();
    } catch (ex) {
      setErr(ex instanceof Error ? ex.message : 'Failed to delete project');
    }
  };

  const createDriveProject = async () => {
    const token = await getValidDriveAccessToken();
    if (!token) {
      alert('Connect Google Drive first.');
      return;
    }
    const name = driveName.trim() || 'Drive dataset';
    const url = driveUrl.trim();
    if (!url) {
      alert('Paste a Google Drive folder link.');
      return;
    }
    setDriveBusy(true);
    setErr(null);
    try {
      const proj = await api.createProjectFromDrive(name, url, token);
      setDriveOpen(false);
      setDriveName('');
      setDriveUrl('');
      await onCreateDrive(proj.id);
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'Failed to create Drive project');
    } finally {
      setDriveBusy(false);
    }
  };

  return (
    <div className="min-h-full flex flex-col bg-ocean-deep text-gray-200">
      <header className="border-b border-ocean-border bg-ocean-card px-6 py-4 flex flex-wrap items-center gap-3 shrink-0">
        <AnnotraBrand size="md" showTagline />
        <span className="text-xs text-gray-500 hidden lg:inline">Projects</span>
        <span className="flex-1" />
        {driveConfigured && (
          <button
            type="button"
            onClick={() => void startDriveOAuth()}
            className={`text-xs px-3 py-1.5 rounded border ${
              driveConnected
                ? 'border-emerald-600/60 text-emerald-300'
                : 'border-ocean-border hover:bg-ocean-deep'
            }`}
          >
            {driveConnected ? 'Google Drive connected' : 'Connect Google Drive'}
          </button>
        )}
        <button
          type="button"
          onClick={() => setDriveOpen(true)}
          disabled={!driveConfigured}
          title={driveConfigured ? undefined : 'Configure MARINE_GOOGLE_CLIENT_ID in backend .env'}
          className="text-xs px-3 py-1.5 rounded border border-ocean-border hover:bg-ocean-deep disabled:opacity-40"
        >
          + From Google Drive
        </button>
        <button
          type="button"
          onClick={() => setJoinOpen(true)}
          className="text-xs px-3 py-1.5 rounded border border-ocean-border hover:bg-ocean-deep"
        >
          Join shared project
        </button>
        <button
          type="button"
          onClick={onCreateNew}
          className="text-xs px-3 py-1.5 rounded bg-ocean-teal text-ocean-deep font-semibold hover:opacity-90"
        >
          + Local folder
        </button>
        <button
          type="button"
          onClick={() => refresh()}
          className="text-xs px-3 py-1.5 rounded border border-ocean-border hover:bg-ocean-deep"
          disabled={loading}
        >
          Refresh
        </button>
      </header>

      <main className="flex-1 p-6 overflow-auto">
        {err && !joinOpen && !driveOpen && (
          <div className="mb-4 text-xs text-amber-300 bg-amber-950/40 border border-amber-800/50 rounded px-3 py-2">
            {err}
          </div>
        )}
        {loading ? (
          <p className="text-sm text-gray-500 animate-pulse">Loading projects…</p>
        ) : projects.length === 0 ? (
          <div className="glass rounded-xl p-8 max-w-lg text-center">
            <p className="text-gray-400 text-sm mb-4">
              Create a project from a local folder or a Google Drive folder link (no local copy required).
            </p>
            <div className="flex flex-wrap gap-2 justify-center">
              <button
                type="button"
                onClick={onCreateNew}
                className="text-sm px-4 py-2 rounded bg-ocean-teal text-ocean-deep font-medium"
              >
                Local folder
              </button>
              {driveConfigured && (
                <button
                  type="button"
                  onClick={() => setDriveOpen(true)}
                  className="text-sm px-4 py-2 rounded border border-ocean-border"
                >
                  Google Drive
                </button>
              )}
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {projects.map((p) => (
              <div
                key={p.id}
                role="button"
                tabIndex={0}
                onClick={() => void onOpenProject(p.id, null)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    void onOpenProject(p.id, null);
                  }
                }}
                className="text-left glass rounded-xl p-4 border border-ocean-border hover:border-ocean-teal/50 hover:bg-ocean-card/80 transition group cursor-pointer relative"
              >
                <button
                  type="button"
                  onClick={(e) => void deleteProject(p, e)}
                  className="absolute top-2 right-2 text-[10px] px-2 py-0.5 rounded border border-red-900/60 text-red-300/90 hover:bg-red-950/50 opacity-0 group-hover:opacity-100 focus:opacity-100 transition"
                  title="Remove project from Annotra"
                >
                  Delete
                </button>
                <div className="flex items-center gap-2 pr-14">
                  <div className="text-ocean-teal font-semibold truncate group-hover:text-white flex-1">
                    {p.name}
                  </div>
                  {isDriveProject(p) && (
                    <span className="text-[9px] uppercase tracking-wide px-1.5 py-0.5 rounded bg-blue-900/50 text-blue-200 shrink-0">
                      Drive
                    </span>
                  )}
                </div>
                <div className="text-[10px] text-gray-500 truncate mt-1 font-mono" title={p.root_path}>
                  {isDriveProject(p) ? `Google Drive · ${p.drive_folder_id ?? p.root_path}` : p.root_path}
                </div>
                <div className="mt-3 flex gap-3 text-xs text-gray-400">
                  <span>{p.image_count} images</span>
                  <span>{p.annotated_count} labeled</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>

      {driveOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/65 p-4">
          <div className="glass rounded-xl p-6 max-w-md w-full border border-ocean-border">
            <h2 className="text-ocean-teal font-semibold mb-3">New project from Google Drive</h2>
            <p className="text-xs text-gray-500 mb-3">
              Paste a shared folder link. Images stay on Drive; annotations are saved to the app database and synced
              to <code className="text-gray-400">.marine-studio/annotations/</code> in that folder.
            </p>
            {!driveConnected && (
              <p className="text-xs text-amber-300 mb-2">Connect Google Drive in the header first.</p>
            )}
            <input
              className="w-full bg-ocean-deep border border-ocean-border rounded px-3 py-2 text-sm mb-2"
              placeholder="Project name"
              value={driveName}
              onChange={(e) => setDriveName(e.target.value)}
            />
            <input
              className="w-full bg-ocean-deep border border-ocean-border rounded px-3 py-2 text-sm mb-2 font-mono text-xs"
              placeholder="https://drive.google.com/drive/folders/..."
              value={driveUrl}
              onChange={(e) => setDriveUrl(e.target.value)}
            />
            {err && driveOpen && <p className="text-xs text-amber-300 mb-2">{err}</p>}
            <div className="flex justify-end gap-2 mt-4">
              <button
                type="button"
                onClick={() => {
                  setDriveOpen(false);
                  setErr(null);
                }}
                className="text-xs px-3 py-1.5 rounded border border-ocean-border"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={driveBusy || !driveConnected}
                onClick={() => void createDriveProject()}
                className="text-xs px-3 py-1.5 rounded bg-ocean-teal text-ocean-deep font-medium disabled:opacity-40"
              >
                {driveBusy ? 'Indexing…' : 'Create & open'}
              </button>
            </div>
          </div>
        </div>
      )}

      {joinOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/65 p-4">
          <div className="glass rounded-xl p-6 max-w-md w-full border border-ocean-border">
            <h2 className="text-ocean-teal font-semibold mb-3">Join shared project</h2>
            <p className="text-xs text-gray-500 mb-3">
              Paste the share code. For Drive projects, connect Google Drive (folder must be shared with you). For local
              projects, pick your matching dataset folder.
            </p>
            <input
              className="w-full bg-ocean-deep border border-ocean-border rounded px-3 py-2 text-sm mb-2 font-mono"
              placeholder="Share code (UUID)"
              value={joinToken}
              onChange={(e) => setJoinToken(e.target.value)}
            />
            {err && joinOpen && <p className="text-xs text-amber-300 mb-2">{err}</p>}
            <div className="flex gap-2 mb-4">
              <button
                type="button"
                disabled={joinBusy}
                onClick={() => lookupShare()}
                className="text-xs px-3 py-1.5 rounded border border-ocean-border hover:bg-ocean-deep"
              >
                Verify code
              </button>
            </div>
            {joinPreview && (
              <div className="text-xs text-gray-300 mb-4 p-2 rounded bg-ocean-deep/80 border border-ocean-border/60">
                <strong className="text-ocean-teal">{joinPreview.name}</strong>
                {isDriveProject(joinPreview) ? (
                  <div className="text-gray-500 mt-1">Google Drive project — connect Drive, then open.</div>
                ) : (
                  <div className="text-gray-500 mt-1 truncate">{joinPreview.root_path}</div>
                )}
                <div className="mt-2">{joinPreview.image_count} images</div>
              </div>
            )}
            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={() => {
                  setJoinOpen(false);
                  setJoinPreview(null);
                  setErr(null);
                }}
                className="text-xs px-3 py-1.5 rounded border border-ocean-border"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={!joinPreview || joinBusy}
                onClick={() => void finishJoin()}
                className="text-xs px-3 py-1.5 rounded bg-ocean-teal text-ocean-deep font-medium disabled:opacity-40"
              >
                {joinPreview && isDriveProject(joinPreview) ? 'Open project' : 'Choose local folder & open'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

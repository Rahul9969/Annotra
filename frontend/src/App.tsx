import { useCallback, useEffect, useState } from 'react';
import { api } from './api';
import { backendOfflineHint, isCloudBrowser, localFolderAvailable } from './apiBase';
import AnnotationCanvas from './components/AnnotationCanvas';
import BatchProgressModal from './components/BatchProgress';
import Dashboard from './components/Dashboard';
import FileTreePanel from './components/FileTreePanel';
import RightPanel from './components/RightPanel';
import ExportPanel from './components/ExportPanel';
import SettingsPanel from './components/SettingsPanel';
import Toolbar from './components/Toolbar';
import { getValidDriveAccessToken, isDriveProject } from './driveAuth';
import { buildFlatNavigationOrder, buildSpeciesGroups, speciesGroupKey } from './components/fileTreeUtils';
import { rawBase64FromDataUrl, resolveImageDisplaySrc, resolveImageDiskPath } from './imagePath';
import { useKeyboard } from './hooks/useKeyboard';
import { useModelHealth } from './hooks/useModelHealth';
import { modelStatusLabel } from './modelStatus';
import { useStore } from './store';
import type { BBox, ImageStatus, ProjectInfo } from './types';

export default function App() {
  const {
    project,
    setProject,
    images,
    setImages,
    currentImageIndex,
    setCurrentIndex,
    setImageSrc,
    setAnnotations,
    annotations,
    selectedId,
    setClasses,
    setAISettings,
    batchJobId,
    setBatchJobId,
    panel,
    setPanel,
    prompts,
    collaborationLocalRoot,
    setCollaborationLocalRoot,
  } = useStore();

  const [route, setRoute] = useState<'dashboard' | 'workspace'>('dashboard');
  const [search, setSearch] = useState('');
  const [treeExpanded, setTreeExpanded] = useState<Set<string>>(() => new Set());
  const [stats, setStats] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const { modelState, modelsLoaded, modelError, modelsReady, backendOk } = useModelHealth();

  useEffect(() => {
    api.getAISettings().then(setAISettings).catch(() => undefined);
  }, [setAISettings]);


  useEffect(() => {
    if (route !== 'workspace' || !project?.id) return;
    const ws = new WebSocket(api.projectWsUrl(project.id));
    ws.onmessage = async (ev) => {
      let msg: { type?: string; image_id?: number };
      try {
        msg = JSON.parse(ev.data as string);
      } catch {
        return;
      }
      if (msg.type !== 'annotations_saved' || typeof msg.image_id !== 'number') return;
      const { currentImage, images, imagesTotal } = useStore.getState();
      if (!currentImage || currentImage.id !== msg.image_id) {
        try {
          const anns = await api.getAnnotations(msg.image_id);
          const idx = images.findIndex((x) => x.id === msg.image_id);
          if (idx >= 0) {
            const next = [...images];
            next[idx] = {
              ...next[idx],
              annotation_count: anns.length,
            };
            useStore.getState().setImages(next, imagesTotal);
          }
        } catch {
          /* ignore */
        }
        return;
      }
      const anns = await api.getAnnotations(msg.image_id);
      setAnnotations(
        (anns as BBox[]).map((a, i) => ({ ...a, id: a.id ?? i + 1 })),
        false,
      );
    };
    return () => ws.close();
  }, [route, project?.id, setAnnotations]);

  const persistCurrent = useCallback(
    async (status?: ImageStatus) => {
      const { currentImage, annotations, images, currentImageIndex, imagesTotal } = useStore.getState();
      if (!currentImage) return;

      let nextStatus: ImageStatus = status ?? currentImage.status;
      if (!status) {
        if (annotations.length === 0) nextStatus = 'unannotated';
        else if (currentImage.status !== 'verified' && currentImage.status !== 'flagged') {
          nextStatus = 'ai';
        }
      }

      await api.saveAnnotations(currentImage.id, annotations, nextStatus);
      const nextImages = images.map((img, i) =>
        i === currentImageIndex
          ? { ...img, status: nextStatus, annotation_count: annotations.length }
          : img,
      );
      setImages(nextImages, imagesTotal);
    },
    [setImages],
  );

  const goDashboard = async () => {
    try {
      await persistCurrent();
    } catch {
      /* ignore */
    }
    setCollaborationLocalRoot(null);
    setRoute('dashboard');
  };

  const shareProject = async () => {
    if (!project) return;
    try {
      const { share_token } = await api.shareProject(project.id);
      await navigator.clipboard.writeText(share_token);
      if (isDriveProject(project)) {
        alert(
          `Share code copied:\n${share_token}\n\nFriend: Dashboard → Join shared project → paste code → Connect Google Drive (same account with access to the folder).\nAnnotations sync via the API and JSON files in Drive (.marine-studio/annotations).`,
        );
      } else {
        alert(
          `Share code copied:\n${share_token}\n\nYour friend: Dashboard → Join shared project → paste this code → select their local copy of the same dataset folder.\nYou must both use the same API server (same PC/LAN).`,
        );
      }
    } catch (e) {
      alert(e instanceof Error ? e.message : 'Share failed');
    }
  };

  const loadImage = useCallback(
    async (index: number) => {
      const list = useStore.getState().images;
      const im = list[index];
      if (!im) return;
      setCurrentIndex(index);
      setLoading(true);
      try {
        const { project: proj, collaborationLocalRoot: collab } = useStore.getState();
        const src = await resolveImageDisplaySrc(im, proj, collab);
        setImageSrc(src);
        const anns = await api.getAnnotations(im.id);
        setAnnotations(
          (anns as BBox[]).map((a, i) => ({ ...a, id: a.id ?? i + 1 })),
          false,
        );
      } finally {
        setLoading(false);
      }
    },
    [setCurrentIndex, setImageSrc, setAnnotations],
  );

  const enterWorkspace = useCallback(
    async (projectId: number, collabRoot: string | null) => {
      setLoading(true);
      setCollaborationLocalRoot(collabRoot);
      setSearch('');
      setImageSrc(null);
      setAnnotations([], false);
      try {
        const refreshed = (await api.getProject(projectId)) as ProjectInfo;
        setProject(refreshed);
        const mirror = collabRoot ?? refreshed.local_mirror_path ?? null;
        setCollaborationLocalRoot(mirror);
        if (isDriveProject(refreshed) && (await getValidDriveAccessToken())) {
          try {
            await api.driveImportAnnotations(projectId, false);
          } catch {
            /* optional sync from Drive sidecars */
          }
        }
        const { items, total } = await api.listAllImages(projectId);
        try {
          await api.reindexSpecies(projectId);
        } catch {
          /* non-fatal */
        }
        const { items: refreshedItems, total: refreshedTotal } = await api.listAllImages(projectId);
        setImages(refreshedItems, refreshedTotal);
        const cls = await api.listClasses(projectId);
        setClasses(cls);
        setRoute('workspace');
        setCurrentIndex(0);
        if (items.length) {
          await loadImage(0);
        }
      } catch (e) {
        alert(e instanceof Error ? e.message : 'Failed to open project');
        setRoute('dashboard');
      } finally {
        setLoading(false);
      }
    },
    [loadImage, setAnnotations, setClasses, setCollaborationLocalRoot, setCurrentIndex, setImages, setImageSrc, setProject],
  );

  const navigateToImage = useCallback(
    async (index: number) => {
      const { currentImageIndex, images } = useStore.getState();
      if (index < 0 || index >= images.length) return;
      if (index !== currentImageIndex && images[currentImageIndex]) {
        try {
          await persistCurrent();
        } catch (e) {
          const msg = e instanceof Error ? e.message : 'Save failed';
          if (!window.confirm(`${msg}\n\nLeave this image without saving?`)) return;
        }
      }
      await loadImage(index);
    },
    [persistCurrent, loadImage],
  );

  const openFolder = async () => {
    if (!localFolderAvailable()) {
      alert(
        isCloudBrowser()
          ? 'Local folders are not available in the cloud app. Create a Google Drive project, or use the Annotra desktop app.'
          : 'Local folders require the Electron desktop app (npm run dev:electron from marine-annotation-studio).',
      );
      return;
    }
    const marine = window.marineAPI!;
    const folder = await marine.openFolder();
    if (!folder) return;

    setLoading(true);
    try {
      const files = await marine.scanFolder(folder, true);
      if (!files.length) {
        alert('No images found in that folder.');
        return;
      }

      const folderClasses = [
        ...new Set(
          files.map((f) => {
            const parts = f.path.replace(/\\/g, '/').split('/');
            return parts.length >= 2 ? parts[parts.length - 2] : '';
          }).filter(Boolean),
        ),
      ];
      const proj = (await api.createProject(
        folder.split(/[/\\]/).pop() ?? 'Dataset',
        folder,
        folderClasses,
      )) as ProjectInfo;

      await api.reindexProject(proj.id, files.map((f) => ({ path: f.path, folder: f.folder })));
      await api.reindexSpecies(proj.id);

      await enterWorkspace(proj.id, null);
    } catch (e) {
      alert(e instanceof Error ? e.message : 'Failed to open folder');
    } finally {
      setLoading(false);
    }
  };

  const importDataset = async () => {
    if (!localFolderAvailable()) {
      alert(
        isCloudBrowser()
          ? 'Importing datasets from disk requires the Annotra desktop app.'
          : 'Dataset import requires the Electron desktop app (npm run dev:electron).',
      );
      return;
    }
    const marine = window.marineAPI!;
    const folder = await marine.openFolder();
    if (!folder) return;

    setLoading(true);
    try {
      const res = await api.importDataset(folder);
      await enterWorkspace(res.project_id, null);
    } catch (e) {
      alert(e instanceof Error ? e.message : 'Failed to import dataset');
    } finally {
      setLoading(false);
    }
  };

  const saveCurrent = async () => {
    try {
      await persistCurrent('verified');
    } catch (e) {
      alert(e instanceof Error ? e.message : 'Save failed');
    }
  };

  const autoAnnotate = async () => {
    const im = useStore.getState().currentImage;
    if (!im || !project) return;
    if (!modelsReady) {
      alert(
        modelStatusLabel(modelState, modelsLoaded, modelError, backendOk) +
          '\n\nAuto stays disabled until all models finish loading.',
      );
      return;
    }
    setLoading(true);
    try {
      const { collaborationLocalRoot: collab, project: proj } = useStore.getState();
      let imageBase64: string | null = null;
      if (collab && window.marineAPI && !isDriveProject(proj)) {
        const diskPath = await resolveImageDiskPath(im, collab);
        const dataUrl = await window.marineAPI.readFileBase64(diskPath);
        imageBase64 = rawBase64FromDataUrl(dataUrl);
      }
      /* Drive images: server downloads via Bearer token (cached locally). */
      const res = await api.annotate(
        im.path,
        im.id,
        project.id,
        prompts.split(',').map((s) => s.trim()),
        imageBase64,
        useStore.getState().aiSettings
      );
      const withIds = res.annotations.map((a, i) => ({
        ...a,
        id: a.id ?? Date.now() + i,
      }));
      setAnnotations(withIds, false);
      if (withIds.length === 0) {
        alert(
          `No boxes detected for "${im.species_class ?? 'unknown'}".\n` +
            `Try drawing a box manually, lowering confidence in Settings, or check the image lighting.`,
        );
      }
    } catch (e) {
      alert(e instanceof Error ? e.message : 'Auto-annotate failed');
    } finally {
      setLoading(false);
    }
  };

  const linkLocalFolder = async () => {
    if (!project || !window.marineAPI) return;
    const folder = await window.marineAPI.openFolder();
    if (!folder) return;
    setLoading(true);
    try {
      const res = await api.setLocalMirror(project.id, folder);
      const path = res.local_mirror_path ?? folder;
      setCollaborationLocalRoot(path);
      setProject({ ...project, local_mirror_path: path });
      alert(
        'Local folder linked.\n\n• Batch & export use your disk (faster)\n• Saves still sync to Google Drive for collaborators',
      );
    } catch (e) {
      alert(e instanceof Error ? e.message : 'Failed to link folder');
    } finally {
      setLoading(false);
    }
  };

  const batchAnnotate = async (pipelineMode?: string) => {
    if (!project) return;
    const mirror = project.local_mirror_path ?? useStore.getState().collaborationLocalRoot;
    if (isDriveProject(project)) {
      const token = await getValidDriveAccessToken();
      if (!token && !mirror) {
        alert('Connect Google Drive or link your local copy of the dataset (toolbar: Link local).');
        return;
      }
    } else if (useStore.getState().collaborationLocalRoot) {
      alert(
        'Batch uses paths on the machine running the backend. Link the dataset on that machine, or run batch on the host PC.',
      );
      return;
    }
    if (!modelsReady) {
      alert(
        modelStatusLabel(modelState, modelsLoaded, modelError, backendOk) +
          '\n\nBatch stays disabled until all models finish loading.',
      );
      return;
    }
    try {
      const { images, imagesTotal } = useStore.getState();
      const unannotated = images.filter((i) => i.status === 'unannotated').length;
      const projectTotal = imagesTotal || images.length;
      const unannotatedOnly = window.confirm(
        `${pipelineMode === 'smart_cv' ? 'Smart ' : ''}Batch: process ${unannotated} unannotated image(s)?\n\n` +
          `OK = unannotated only\nCancel = ALL ${projectTotal} images in this project`,
      );
      const { job_id, total } = await api.batchStart(
        project.id,
        undefined,
        unannotatedOnly,
        !unannotatedOnly,
      );
      if (total === 0) {
        alert(
          unannotatedOnly
            ? 'No unannotated images left. Run batch again and choose Cancel to re-annotate all images.'
            : 'No images in this project. Re-open your folder or re-index Drive.',
        );
        return;
      }
      setBatchJobId(job_id);
    } catch (e) {
      alert(e instanceof Error ? e.message : 'Batch failed');
    }
  };

  const smartBatchAnnotate = () => batchAnnotate('smart_cv');

  const openExport = () => {
    if (!project) {
      alert('Open a dataset folder first.');
      return;
    }
    setPanel('export');
  };

  const showStats = async () => {
    if (!project) return;
    setStats(await api.stats(project.id));
    setPanel('stats');
  };

  const refreshDriveIndex = async () => {
    if (!project || !isDriveProject(project)) return;
    setLoading(true);
    try {
      await api.reindexDriveProject(project.id);
      const { items, total } = await api.listAllImages(project.id);
      setImages(items, total);
      alert('Drive folder re-indexed.');
    } catch (e) {
      alert(e instanceof Error ? e.message : 'Re-index failed');
    } finally {
      setLoading(false);
    }
  };

  const importFromDrive = async () => {
    if (!project || !isDriveProject(project)) return;
    const overwrite = window.confirm(
      'Import annotation JSON from Google Drive?\n\nOK = overwrite DB when Drive is newer\nCancel = only fill empty images',
    );
    setLoading(true);
    try {
      const res = await api.driveImportAnnotations(project.id, overwrite);
      const { items, total } = await api.listAllImages(project.id);
      setImages(items, total);
      const idx = useStore.getState().currentImageIndex;
      if (items[idx]) await loadImage(idx);
      alert(`Imported ${res.imported} image(s) from Drive (${res.skipped} skipped, ${res.files} sidecar files).`);
    } catch (e) {
      alert(e instanceof Error ? e.message : 'Import failed');
    } finally {
      setLoading(false);
    }
  };

  const clearDriveCache = async () => {
    if (!project || !isDriveProject(project)) return;
    if (!window.confirm('Clear local image cache for this project? Images will re-download from Drive.')) return;
    try {
      const res = await api.driveClearCache(project.id);
      const mb = (res.bytes_freed / (1024 * 1024)).toFixed(1);
      alert(`Cache cleared (~${mb} MB freed).`);
    } catch (e) {
      alert(e instanceof Error ? e.message : 'Clear cache failed');
    }
  };

  useEffect(() => {
    const cur = images[currentImageIndex];
    if (!cur) return;
    const key = speciesGroupKey(cur);
    setTreeExpanded((prev) => {
      if (prev.has(key)) return prev;
      const next = new Set(prev);
      next.add(key);
      return next;
    });
  }, [currentImageIndex, images]);

  const goPrev = () => {
    const { images: imgs, currentImageIndex: cur } = useStore.getState();
    const treeOrder = buildFlatNavigationOrder(imgs, search, treeExpanded);
    const pos = treeOrder.indexOf(cur);
    const target = pos > 0 ? treeOrder[pos - 1] : cur > 0 ? cur - 1 : null;
    if (target != null) void navigateToImage(target);
  };

  const goNext = () => {
    const { images: imgs, currentImageIndex: cur } = useStore.getState();
    const treeOrder = buildFlatNavigationOrder(imgs, search, treeExpanded);
    const pos = treeOrder.indexOf(cur);
    const target =
      pos >= 0 && pos < treeOrder.length - 1
        ? treeOrder[pos + 1]
        : cur < imgs.length - 1
          ? cur + 1
          : null;
    if (target != null) void navigateToImage(target);
  };

  const deleteSelected = () => {
    if (selectedId == null) return;
    setAnnotations(annotations.filter((a, i) => (a.id ?? i) !== selectedId));
  };

  useKeyboard({
    onPrev: goPrev,
    onNext: goNext,
    onSave: saveCurrent,
    onAutoAnnotate: autoAnnotate,
    onBatchAnnotate: batchAnnotate,
    onFit: () => undefined,
    onDelete: deleteSelected,
    modelsReady,
  });

  if (route === 'dashboard') {
    return (
      <div className="h-full flex flex-col bg-ocean-deep">
        {!backendOk && (
          <div className="bg-amber-900/40 text-amber-200 text-xs px-4 py-2 text-center shrink-0">
            {backendOfflineHint()}
          </div>
        )}
        <div className="flex-1 min-h-0">
          <Dashboard
            onOpenProject={enterWorkspace}
            onCreateNew={openFolder}
            onImportDataset={importDataset}
            onCreateDrive={async (projectId) => enterWorkspace(projectId, null)}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <Toolbar
        onOpenFolder={openFolder}
        onDashboard={goDashboard}
        onShareProject={shareProject}
        isDriveProject={isDriveProject(project)}
        onLinkLocalFolder={linkLocalFolder}
        localMirrorLinked={Boolean(project?.local_mirror_path ?? collaborationLocalRoot)}
        onRefreshDrive={refreshDriveIndex}
        onImportFromDrive={importFromDrive}
        onClearDriveCache={clearDriveCache}
        onSave={saveCurrent}
        onAutoAnnotate={autoAnnotate}
        onBatchAnnotate={() => batchAnnotate()}
        onSmartBatch={smartBatchAnnotate}
        onExport={openExport}
        onSettings={() => setPanel('settings')}
        onStats={showStats}
        onPrev={goPrev}
        onNext={goNext}
        modelsReady={modelsReady}
        modelState={modelState}
        modelsLoaded={modelsLoaded}
        modelError={modelError}
        backendOk={backendOk}
      />

      {!modelsReady && (
        <div
          className={`text-xs px-4 py-2 text-center border-b border-ocean-border shrink-0 ${
            modelState === 'error' ? 'bg-amber-950/50 text-amber-200' : 'bg-ocean-blue/25 text-ocean-teal'
          }`}
        >
          <span className={modelState === 'loading' || modelState === 'not_loaded' ? 'animate-pulse' : ''}>
            {modelStatusLabel(modelState, modelsLoaded, modelError, backendOk)} — Auto and Batch are disabled
            until loading completes.
          </span>
        </div>
      )}

      <div className="flex flex-1 min-h-0">
        <aside className="w-64 border-r border-ocean-border bg-ocean-card flex flex-col shrink-0">
          <div className="p-3 border-b border-ocean-border">
            <input
              type="search"
              placeholder="Search files…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full bg-ocean-deep border border-ocean-border rounded px-2 py-1 text-xs"
            />
          </div>

          {project && stats === null && (
            <div className="p-3 text-xs text-gray-400 glass m-2 rounded-lg">
              <div className="text-ocean-teal font-semibold mb-1">{project.name}</div>
              <div>{project.image_count} images</div>
              <div>{project.annotated_count} annotated</div>
            </div>
          )}

          <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
            <FileTreePanel
              search={search}
              expanded={treeExpanded}
              onExpandedChange={setTreeExpanded}
              onSelect={(i) => void navigateToImage(i)}
            />
          </div>
        </aside>

        <main className="flex-1 flex flex-col min-w-0 relative">
          {loading && (
            <div className="absolute inset-0 z-10 flex items-center justify-center bg-black/40">
              <span className="text-ocean-teal animate-pulse">Processing…</span>
            </div>
          )}
          <AnnotationCanvas />
        </main>

        <RightPanel />
      </div>

      {panel === 'settings' && <SettingsPanel onClose={() => setPanel(null)} />}

      {panel === 'export' && project && (
        <ExportPanel
          projectId={project.id}
          projectName={project.name}
          projectSource={project.source}
          localMirrorPath={project.local_mirror_path ?? collaborationLocalRoot}
          onClose={() => setPanel(null)}
        />
      )}

      {panel === 'stats' && stats && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="glass rounded-xl p-6 max-w-md">
            <h2 className="text-ocean-teal font-semibold mb-3">Dataset Statistics</h2>
            <pre className="text-xs font-mono text-gray-300 overflow-auto max-h-96">
              {JSON.stringify(stats, null, 2)}
            </pre>
            <button type="button" className="mt-4 px-4 py-2 rounded border border-ocean-border" onClick={() => setPanel(null)}>
              Close
            </button>
          </div>
        </div>
      )}

      {batchJobId && project && (
        <BatchProgressModal
          jobId={batchJobId}
          onClose={() => setBatchJobId(null)}
          onComplete={async () => {
            const { items, total } = await api.listAllImages(project.id);
            setImages(items, total);
            const refreshed = (await api.getProject(project.id)) as ProjectInfo;
            setProject(refreshed);
            const idx = useStore.getState().currentImageIndex;
            if (items.length && idx >= 0 && idx < items.length) await loadImage(idx);
          }}
        />
      )}

    </div>
  );
}

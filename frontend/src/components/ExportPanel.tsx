import { useState } from 'react';
import { api } from '../api';
import { isDriveProject } from '../driveAuth';

const FORMATS: { id: string; label: string; hint: string }[] = [
  { id: 'yolo_v11_box', label: 'YOLOv11 Object Detection (Bounding Box)', hint: 'Standard YOLO txt labels; train with yolo11n.pt' },
  { id: 'yolo_v11_seg', label: 'YOLOv11 Instance Segmentation (Polygon)', hint: 'YOLO txt labels with polygon coords; train with yolo11n-seg.pt' },
  { id: 'yolo_v8_box', label: 'YOLOv8 Object Detection (Bounding Box)', hint: 'Standard YOLO txt labels; train with yolov8n.pt' },
  { id: 'yolo_v8_seg', label: 'YOLOv8 Instance Segmentation (Polygon)', hint: 'YOLO txt labels with polygon coords; train with yolov8n-seg.pt' },
  { id: 'yolo_v5', label: 'YOLOv5 Object Detection', hint: 'Classic YOLO txt labels' },
  { id: 'coco', label: 'COCO JSON', hint: 'instances.json' },
  { id: 'voc', label: 'Pascal VOC XML', hint: 'One XML per image' },
  { id: 'csv', label: 'CSV spreadsheet', hint: 'Flat table of all boxes' },
  { id: 'labeled_images', label: 'Annotated preview images only', hint: 'JPEG copies with boxes drawn' },
];

export default function ExportPanel({
  projectId,
  projectName,
  projectSource,
  localMirrorPath,
  onClose,
}: {
  projectId: number;
  projectName: string;
  projectSource?: string;
  localMirrorPath?: string | null;
  onClose: () => void;
}) {
  const [format, setFormat] = useState('yolo_v11_box');
  const [reviewedOnly, setReviewedOnly] = useState(false);
  const [includePreviews, setIncludePreviews] = useState(true);
  const [createZip, setCreateZip] = useState(true);
  const [splitTrain, setSplitTrain] = useState(0.8);
  const [splitVal, setSplitVal] = useState(0.1);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const runExport = async () => {
    if (!window.marineAPI) {
      alert('Export requires the Electron desktop app.');
      return;
    }
    const parentDir = await window.marineAPI.saveFolder();
    if (!parentDir) return;

    setBusy(true);
    setResult(null);
    try {
      const res = await api.export({
        project_id: projectId,
        output_dir: parentDir,
        format,
        split_train: splitTrain,
        split_val: splitVal,
        reviewed_only: reviewedOnly,
        include_labeled_previews: includePreviews && format !== 'labeled_images',
        local_mirror_path: localMirrorPath ?? undefined,
        create_zip: createZip,
      });
      const stats = res.source_stats;
      let speedNote = '';
      if (stats) {
        const local = (stats.local_mirror ?? 0) + (stats.local_path ?? 0);
        const drive = stats.drive_cache ?? 0;
        if (local > 0 && drive === 0) {
          speedNote = '\n\nUsed your local dataset folder (fast).';
        } else if (local > 0 && drive > 0) {
          speedNote = `\n\n${local} images from local disk, ${drive} from Google Drive cache.`;
        } else if (drive > 0) {
          speedNote = '\n\nImages loaded from Google Drive (link local folder for faster export next time).';
        }
      }
      const zipLine = res.zip_path ? `\n\nZIP download:\n${res.zip_path}` : '';
      setResult(
        `Exported ${res.total_exported ?? 0} annotated images to:\n${res.output_dir}${zipLine}\n\n` +
          `Train: ${res.counts?.train ?? 0} · Val: ${res.counts?.val ?? 0} · Test: ${res.counts?.test ?? 0}` +
          speedNote,
      );
      if (res.zip_path) {
        await window.marineAPI.showItemInFolder(res.zip_path);
      } else {
        await window.marineAPI.showItemInFolder(res.output_dir);
      }
    } catch (e) {
      alert(e instanceof Error ? e.message : 'Export failed');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <div className="glass rounded-xl w-full max-w-lg max-h-[90vh] overflow-y-auto p-6">
        <h2 className="text-ocean-teal font-semibold text-lg mb-1">Export annotated dataset</h2>
        <p className="text-xs text-gray-400 mb-4">
          Downloads a <strong className="text-gray-300">new folder</strong> (and optional ZIP) with images + labels for
          every annotated image.
          <br />
          <span className="text-gray-500">{projectName}</span>
        </p>

        {isDriveProject({ source: projectSource }) && (
          <div className="rounded-lg bg-ocean-deep/60 border border-ocean-border p-3 text-xs text-gray-400 mb-4">
            {localMirrorPath ? (
              <p>
                <strong className="text-emerald-300">Local folder linked</strong> — export and batch use disk files
                (faster). Saves still sync to Google Drive for collaborators.
              </p>
            ) : (
              <p>
                <strong className="text-amber-200">Tip:</strong> Use toolbar <em>Link local folder</em> if you have the
                same dataset on this PC. Export and batch will be much faster.
              </p>
            )}
          </div>
        )}

        <label className="block text-xs text-gray-400 mb-1">Format</label>
        <select
          className="w-full bg-ocean-deep border border-ocean-border rounded px-3 py-2 text-sm mb-1"
          value={format}
          onChange={(e) => setFormat(e.target.value)}
        >
          {FORMATS.map((f) => (
            <option key={f.id} value={f.id}>
              {f.label}
            </option>
          ))}
        </select>
        <p className="text-[10px] text-gray-500 mb-4">{FORMATS.find((f) => f.id === format)?.hint}</p>

        <div className="space-y-3 text-sm mb-4">
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={reviewedOnly} onChange={(e) => setReviewedOnly(e.target.checked)} />
            Verified images only
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={includePreviews}
              onChange={(e) => setIncludePreviews(e.target.checked)}
              disabled={format === 'labeled_images'}
            />
            Also save annotated preview images (separate subfolder)
          </label>
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={createZip} onChange={(e) => setCreateZip(e.target.checked)} />
            Create ZIP archive for easy download / sharing
          </label>
        </div>

        {format !== 'labeled_images' && (
          <div className="grid grid-cols-2 gap-3 mb-4 text-xs">
            <label>
              Train split
              <input
                type="number"
                min={0.5}
                max={0.95}
                step={0.05}
                className="w-full mt-1 bg-ocean-deep border border-ocean-border rounded px-2 py-1"
                value={splitTrain}
                onChange={(e) => setSplitTrain(+e.target.value)}
              />
            </label>
            <label>
              Val split
              <input
                type="number"
                min={0.05}
                max={0.3}
                step={0.05}
                className="w-full mt-1 bg-ocean-deep border border-ocean-border rounded px-2 py-1"
                value={splitVal}
                onChange={(e) => setSplitVal(+e.target.value)}
              />
            </label>
          </div>
        )}

        {result && (
          <pre className="text-xs text-gray-300 bg-ocean-deep rounded p-3 mb-4 whitespace-pre-wrap font-mono">
            {result}
          </pre>
        )}

        <div className="flex gap-2 justify-end">
          <button type="button" className="px-4 py-2 rounded border border-ocean-border text-sm" onClick={onClose}>
            Close
          </button>
          <button
            type="button"
            disabled={busy}
            className="px-4 py-2 rounded bg-ocean-teal/20 border border-ocean-teal text-ocean-teal text-sm disabled:opacity-50"
            onClick={runExport}
          >
            {busy ? 'Exporting…' : 'Choose folder & download'}
          </button>
        </div>
      </div>
    </div>
  );
}

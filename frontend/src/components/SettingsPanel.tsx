import { useStore } from '../store';
import { api } from '../api';

export default function SettingsPanel({ onClose }: { onClose: () => void }) {
  const { aiSettings, setAISettings } = useStore();

  const save = async () => {
    await api.updateAISettings(aiSettings);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="glass rounded-xl w-[480px] max-h-[90vh] overflow-y-auto p-6">
        <h2 className="text-lg font-semibold text-ocean-teal mb-4">AI Engine Settings</h2>

        <div className="space-y-4 text-sm">
          <label className="block">
            Confidence ({aiSettings.confidence.toFixed(2)})
            <input
              type="range"
              min={0.1}
              max={1}
              step={0.05}
              value={aiSettings.confidence}
              onChange={(e) => setAISettings({ ...aiSettings, confidence: +e.target.value })}
              className="w-full accent-ocean-teal"
            />
          </label>
          <label className="block">
            IoU Threshold ({aiSettings.iou_threshold.toFixed(2)})
            <input
              type="range"
              min={0.1}
              max={0.95}
              step={0.05}
              value={aiSettings.iou_threshold}
              onChange={(e) => setAISettings({ ...aiSettings, iou_threshold: +e.target.value })}
              className="w-full accent-ocean-teal"
            />
          </label>
          <label className="block">
            Max boxes / image
            <input
              type="number"
              value={aiSettings.max_boxes}
              onChange={(e) => setAISettings({ ...aiSettings, max_boxes: +e.target.value })}
              className="w-full mt-1 bg-ocean-deep border border-ocean-border rounded px-2 py-1"
            />
          </label>
          <label className="block">
            Worker threads
            <input
              type="number"
              min={1}
              max={16}
              value={aiSettings.thread_pool_size}
              onChange={(e) => setAISettings({ ...aiSettings, thread_pool_size: +e.target.value })}
              className="w-full mt-1 bg-ocean-deep border border-ocean-border rounded px-2 py-1"
            />
          </label>
          <label className="block">
            Batch size
            <input
              type="number"
              value={aiSettings.batch_size}
              onChange={(e) => setAISettings({ ...aiSettings, batch_size: +e.target.value })}
              className="w-full mt-1 bg-ocean-deep border border-ocean-border rounded px-2 py-1"
            />
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={aiSettings.enable_sam}
              onChange={(e) => setAISettings({ ...aiSettings, enable_sam: e.target.checked })}
            />
            Refine with SAM2 Segmentation
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={aiSettings.half_precision}
              onChange={(e) => setAISettings({ ...aiSettings, half_precision: e.target.checked })}
            />
            FP16 half precision (GPU)
          </label>
        </div>

        <div className="flex justify-end gap-2 mt-6">
          <button type="button" onClick={onClose} className="px-4 py-2 rounded border border-ocean-border">
            Cancel
          </button>
          <button type="button" onClick={save} className="px-4 py-2 rounded bg-ocean-blue text-white">
            Save
          </button>
        </div>
      </div>
    </div>
  );
}

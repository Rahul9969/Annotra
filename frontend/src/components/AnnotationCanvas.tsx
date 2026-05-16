import { useCallback, useEffect, useRef, useState } from 'react';
import { Circle, Group, Image as KonvaImage, Layer, Line, Rect, Stage, Text, Transformer } from 'react-konva';
import type Konva from 'konva';
import { api } from '../api';
import { bboxFromPolygon, buildSegmentImagePayload } from '../segmentPayload';
import { useStore } from '../store';
import type { BBox } from '../types';

function useImage(src: string | null) {
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    if (!src) {
      setImg(null);
      return;
    }
    const el = new window.Image();
    el.onload = () => setImg(el);
    el.src = src;
  }, [src]);
  return img;
}

function toolCursor(tool: string): string {
  if (tool === 'box' || tool === 'polygon' || tool === 'magic' || tool === 'smart') return 'crosshair';
  if (tool === 'pan') return 'grab';
  return 'default';
}

function isSegmentSource(source?: string): boolean {
  if (!source) return false;
  return /magic|sam|smart|polygon|segment|grabcut/i.test(source);
}

function normalizePolygon(raw: unknown): number[][] | undefined {
  if (!Array.isArray(raw) || raw.length < 3) return undefined;
  const pts: number[][] = [];
  for (const p of raw) {
    if (Array.isArray(p) && p.length >= 2) {
      pts.push([Number(p[0]), Number(p[1])]);
    }
  }
  return pts.length >= 3 ? pts : undefined;
}

function hexToRgba(hex: string, alpha: number): string {
  const h = hex.replace('#', '');
  if (h.length !== 6) return `rgba(0, 245, 212, ${alpha})`;
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export default function AnnotationCanvas() {
  const {
    imageSrc,
    annotations,
    selectedId,
    tool,
    zoom,
    pan,
    showLabels,
    showBoxes,
    classes,
    project,
    currentImage,
    collaborationLocalRoot,
    setAnnotations,
    selectAnnotation,
    setZoom,
    setPan,
  } = useStore();

  const img = useImage(imageSrc);
  const stageRef = useRef<Konva.Stage>(null);
  const trRef = useRef<Konva.Transformer>(null);
  const [drawing, setDrawing] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const [polyPoints, setPolyPoints] = useState<number[][]>([]);
  const [segmentBusy, setSegmentBusy] = useState(false);
  const [stageSize, setStageSize] = useState({ w: 800, h: 600 });
  const containerRef = useRef<HTMLDivElement>(null);
  const isPanning = useRef(false);
  const lastPan = useRef({ x: 0, y: 0 });
  const spaceDown = useRef(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      spaceDown.current = e.code === 'Space' && e.type === 'keydown';
    };
    window.addEventListener('keydown', onKey);
    window.addEventListener('keyup', onKey);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('keyup', onKey);
    };
  }, []);

  useEffect(() => {
    setPolyPoints([]);
  }, [tool, imageSrc]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === 'Escape') {
        setPolyPoints([]);
        setDrawing(null);
      }
      if (e.key === 'Enter' && tool === 'polygon' && polyPoints.length >= 3) {
        finishPolygon();
      }
      if (e.key === 'Backspace' && tool === 'polygon' && polyPoints.length > 0) {
        setPolyPoints((pts) => pts.slice(0, -1));
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tool, polyPoints]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setStageSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const stage = stageRef.current;
    const tr = trRef.current;
    if (!stage || !tr) return;
    const node = stage.findOne(`#ann-${selectedId}`);
    tr.nodes(node ? [node] : []);
    tr.getLayer()?.batchDraw();
  }, [selectedId, annotations]);

  const classColor = (name: string) => classes.find((c) => c.name === name)?.color ?? '#FF2D95';
  const defaultClass = () => {
    const fromList = classes[0]?.name;
    if (fromList) return fromList;
    const fromAnn = useStore.getState().annotations[0]?.class_name;
    if (fromAnn && fromAnn !== 'unknown') return fromAnn;
    return 'fish';
  };

  const fromStage = useCallback(
    (sx: number, sy: number) => ({ x: (sx - pan.x) / zoom, y: (sy - pan.y) / zoom }),
    [zoom, pan],
  );

  const addSegmentAnnotation = (res: {
    polygon: number[][];
    x: number;
    y: number;
    w: number;
    h: number;
    source: string;
  }) => {
    const polygon = normalizePolygon(res.polygon);
    const ann: BBox = {
      id: Date.now(),
      class_name: defaultClass(),
      confidence: 1,
      x: res.x,
      y: res.y,
      w: res.w,
      h: res.h,
      rotation: 0,
      polygon,
      source: res.source,
    };
    const prev = useStore.getState().annotations;
    const withoutAuto = prev.filter((a) => !isSegmentSource(a.source) || a.locked);
    setAnnotations([...withoutAuto, ann]);
    selectAnnotation(ann.id!);
  };

  const finishPolygon = () => {
    if (polyPoints.length < 3) return;
    const box = bboxFromPolygon(polyPoints);
    if (box.w < 4 || box.h < 4) {
      setPolyPoints([]);
      return;
    }
    const ann: BBox = {
      id: Date.now(),
      class_name: defaultClass(),
      confidence: 1,
      ...box,
      rotation: 0,
      polygon: polyPoints.map((p) => [...p]),
      source: 'polygon',
    };
    setAnnotations([...useStore.getState().annotations, ann]);
    selectAnnotation(ann.id!);
    setPolyPoints([]);
  };

  const runMagic = async (ix: number, iy: number) => {
    if (!currentImage) {
      alert('No image loaded');
      return;
    }
    setSegmentBusy(true);
    try {
      const base = await buildSegmentImagePayload(currentImage, project, collaborationLocalRoot);
      const res = await api.segmentMagic({ ...base, x: ix, y: iy });
      addSegmentAnnotation({ ...res, polygon: res.polygon ?? [] });
    } catch (e) {
      alert(e instanceof Error ? e.message : 'Magic wand failed');
    } finally {
      setSegmentBusy(false);
    }
  };

  const runSmart = async (ix: number, iy: number, label: number) => {
    if (!currentImage) {
      alert('No image loaded');
      return;
    }
    setSegmentBusy(true);
    try {
      const base = await buildSegmentImagePayload(currentImage, project, collaborationLocalRoot);
      const res = await api.segmentSmart({
        ...base,
        points: [[ix, iy]],
        labels: [label],
      });
      addSegmentAnnotation({ ...res, polygon: res.polygon ?? [] });
    } catch (e) {
      alert(e instanceof Error ? e.message : 'Smart segment failed');
    } finally {
      setSegmentBusy(false);
    }
  };

  const handleWheel = (e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    setZoom(e.evt.deltaY < 0 ? zoom * 1.08 : zoom / 1.08);
  };

  const onMouseDown = (e: Konva.KonvaEventObject<MouseEvent>) => {
    if (segmentBusy) return;
    if (spaceDown.current || tool === 'pan') {
      isPanning.current = true;
      lastPan.current = { x: e.evt.clientX, y: e.evt.clientY };
      return;
    }
    const stage = e.target.getStage();
    if (!stage) return;
    const ptr = stage.getPointerPosition();
    if (!ptr) return;
    const pos = fromStage(ptr.x, ptr.y);

    if (tool === 'magic') {
      void runMagic(pos.x, pos.y);
      return;
    }
    if (tool === 'smart') {
      const label = e.evt.shiftKey ? 0 : 1;
      void runSmart(pos.x, pos.y, label);
      return;
    }
    if (tool === 'polygon') {
      if (e.evt.detail === 2 && polyPoints.length >= 3) {
        finishPolygon();
        return;
      }
      setPolyPoints((pts) => [...pts, [pos.x, pos.y]]);
      return;
    }
    if (tool !== 'box') return;
    setDrawing({ x: pos.x, y: pos.y, w: 0, h: 0 });
  };

  const onMouseMove = (e: Konva.KonvaEventObject<MouseEvent>) => {
    if (isPanning.current) {
      const dx = e.evt.clientX - lastPan.current.x;
      const dy = e.evt.clientY - lastPan.current.y;
      lastPan.current = { x: e.evt.clientX, y: e.evt.clientY };
      setPan({ x: pan.x + dx, y: pan.y + dy });
      return;
    }
    if (!drawing) return;
    const ptr = e.target.getStage()?.getPointerPosition();
    if (!ptr) return;
    const pos = fromStage(ptr.x, ptr.y);
    setDrawing({ ...drawing, w: pos.x - drawing.x, h: pos.y - drawing.y });
  };

  const onMouseUp = () => {
    isPanning.current = false;
    if (!drawing) return;
    const w = Math.abs(drawing.w);
    const h = Math.abs(drawing.h);
    if (w > 4 && h > 4) {
      setAnnotations([
        ...useStore.getState().annotations,
        {
          id: Date.now(),
          class_name: defaultClass(),
          confidence: 1,
          x: drawing.w < 0 ? drawing.x + drawing.w : drawing.x,
          y: drawing.h < 0 ? drawing.y + drawing.h : drawing.y,
          w,
          h,
          rotation: 0,
          source: 'human',
        },
      ]);
    }
    setDrawing(null);
  };

  const imgW = img?.width ?? 1;
  const imgH = img?.height ?? 1;

  const fitToScreen = () => {
    if (!img) return;
    const scale = Math.min((stageSize.w - 40) / imgW, (stageSize.h - 40) / imgH, 1);
    setZoom(scale);
    setPan({ x: (stageSize.w - imgW * scale) / 2, y: (stageSize.h - imgH * scale) / 2 });
  };

  useEffect(() => {
    if (img) fitToScreen();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [img?.src]);

  const polyStagePoints = polyPoints.flatMap((p) => [pan.x + p[0] * zoom, pan.y + p[1] * zoom]);

  const toolHint =
    tool === 'polygon'
      ? 'Click corners · Enter or double-click to close · Esc cancel'
      : tool === 'magic'
        ? 'Click inside the fish to flood-fill'
        : tool === 'smart'
          ? 'Click fish (foreground) · Shift+click background to refine'
          : null;

  return (
    <div ref={containerRef} className="relative flex-1 bg-[#0a0e14] overflow-hidden">
      <Stage
        ref={stageRef}
        width={stageSize.w}
        height={stageSize.h}
        onWheel={handleWheel}
        className={toolCursor(tool)}
      >
        <Layer onMouseDown={onMouseDown} onMouseMove={onMouseMove} onMouseUp={onMouseUp}>
          {img && (
            <KonvaImage
              image={img}
              x={pan.x}
              y={pan.y}
              width={imgW * zoom}
              height={imgH * zoom}
              listening={tool === 'magic' || tool === 'smart' || tool === 'polygon' || tool === 'box'}
            />
          )}
          {showBoxes &&
            annotations.map((a, i) => {
              if (a.hidden) return null;
              const id = a.id ?? i;
              const color = classColor(a.class_name);
              const selected = selectedId === id;
              const bw = a.w * zoom;
              const bh = a.h * zoom;
              const labelPad = 4;
              const labelW = Math.max(72, a.class_name.length * 6.5 + labelPad * 2);
              const poly = normalizePolygon(a.polygon);
              const polyPts =
                poly && poly.length >= 3
                  ? poly.flatMap((p) => [pan.x + p[0] * zoom, pan.y + p[1] * zoom])
                  : null;
              const strokeW = Math.max(2, 2.5 / Math.max(zoom, 0.15));
              return (
                <Group key={id}>
                  {polyPts && (
                    <>
                      <Line
                        points={polyPts}
                        closed
                        fill={hexToRgba(color, selected ? 0.42 : 0.32)}
                        stroke={hexToRgba(color, 0.95)}
                        strokeWidth={strokeW}
                        lineJoin="round"
                        listening={false}
                      />
                      <Line
                        points={polyPts}
                        closed
                        stroke="#ffffff"
                        strokeWidth={1}
                        opacity={0.35}
                        listening={false}
                      />
                    </>
                  )}
                  <Group
                    id={`ann-${id}`}
                    x={pan.x + a.x * zoom}
                    y={pan.y + a.y * zoom}
                    draggable={tool === 'select' && !a.locked}
                    onClick={() => selectAnnotation(id)}
                  >
                  <Rect
                    width={bw}
                    height={bh}
                    stroke={color}
                    strokeWidth={selected ? 3 : 2}
                    fillEnabled={false}
                    shadowColor={color}
                    shadowBlur={selected ? 8 : 4}
                    shadowOpacity={0.35}
                  />
                  {showLabels && (
                    <Group x={bw + 6} y={Math.max(0, bh / 2 - 9)}>
                      <Rect
                        width={labelW}
                        height={18}
                        fill="rgba(13, 17, 23, 0.88)"
                        cornerRadius={3}
                        stroke={color}
                        strokeWidth={1}
                      />
                      <Text
                        x={labelPad}
                        y={3}
                        text={a.class_name}
                        fontSize={11}
                        fontStyle="bold"
                        fill={color}
                        listening={false}
                      />
                    </Group>
                  )}
                  </Group>
                </Group>
              );
            })}
          {polyPoints.length > 0 && (
            <>
              <Line
                points={polyStagePoints}
                stroke="#00F5D4"
                strokeWidth={2}
                dash={[6, 4]}
                closed={false}
              />
              {polyPoints.map((p, idx) => (
                <Circle
                  key={idx}
                  x={pan.x + p[0] * zoom}
                  y={pan.y + p[1] * zoom}
                  radius={4}
                  fill="#00F5D4"
                  stroke="#0a0e14"
                  strokeWidth={1}
                />
              ))}
            </>
          )}
          {drawing && (
            <Rect
              x={pan.x + (drawing.w < 0 ? drawing.x + drawing.w : drawing.x) * zoom}
              y={pan.y + (drawing.h < 0 ? drawing.y + drawing.h : drawing.y) * zoom}
              width={Math.abs(drawing.w) * zoom}
              height={Math.abs(drawing.h) * zoom}
              stroke="#00F5D4"
              dash={[4, 4]}
            />
          )}
          <Transformer ref={trRef} rotateEnabled />
        </Layer>
      </Stage>

      {toolHint && (
        <div className="absolute top-3 left-3 glass rounded px-2 py-1 text-[10px] text-ocean-teal max-w-xs">
          {toolHint}
        </div>
      )}

      {segmentBusy && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/30 pointer-events-none">
          <span className="text-sm text-ocean-teal animate-pulse">Segmenting…</span>
        </div>
      )}

      <div className="absolute bottom-3 right-3 glass rounded-lg px-3 py-2 text-xs font-mono text-ocean-teal">
        {Math.round(zoom * 100)}%
      </div>

      {imageSrc && (
        <div className="absolute top-3 right-3 glass rounded-lg overflow-hidden w-24 h-16 opacity-70">
          <img src={imageSrc} alt="minimap" className="w-full h-full object-contain" />
        </div>
      )}

      <button
        type="button"
        onClick={fitToScreen}
        className="absolute bottom-3 left-3 glass rounded px-2 py-1 text-xs hover:bg-ocean-border"
      >
        Fit (F)
      </button>
    </div>
  );
}

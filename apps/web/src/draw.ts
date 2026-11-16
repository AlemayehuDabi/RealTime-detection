import type { Detection } from "./types";

const hueCache = new Map<number, string>();

function colorForClass(classId: number): string {
  const cached = hueCache.get(classId);
  if (cached) return cached;
  const hue = (classId * 47) % 360;
  const color = `hsl(${hue} 85% 55%)`;
  hueCache.set(classId, color);
  return color;
}

export function drawDetections(
  ctx: CanvasRenderingContext2D,
  detections: Detection[],
  scaleX: number,
  scaleY: number,
) {
  ctx.lineWidth = 2;
  ctx.font = "14px system-ui, sans-serif";

  for (const det of detections) {
    const { x1, y1, x2, y2 } = det.box;
    const sx1 = x1 * scaleX;
    const sy1 = y1 * scaleY;
    const sx2 = x2 * scaleX;
    const sy2 = y2 * scaleY;
    const color = colorForClass(det.class_id);

    ctx.strokeStyle = color;
    ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1);

    const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
    const textW = ctx.measureText(label).width;
    const ty = Math.max(16, sy1);
    ctx.fillStyle = color;
    ctx.fillRect(sx1, ty - 16, textW + 8, 18);
    ctx.fillStyle = "#111";
    ctx.fillText(label, sx1 + 4, ty - 3);
  }
}
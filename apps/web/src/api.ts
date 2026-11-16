import { getSessionId } from "./session";
import type { DetectionResult } from "./types";

export async function detectImage(blob: Blob): Promise<DetectionResult> {
  const form = new FormData();
  form.append("file", blob, "frame.jpg");

  const res = await fetch("/api/detect", {
    method: "POST",
    body: form,
    headers: {
      "X-Session-Id": getSessionId(),
    },
  });

  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `Detect failed (${res.status})`);
  }

  return (await res.json()) as DetectionResult;
}
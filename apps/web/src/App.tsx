import { useCallback, useEffect, useRef, useState } from "react";
import { detectImage } from "./api";
import { drawDetections } from "./draw";
import type { DetectionResult } from "./types";

type SourceMode = "idle" | "file" | "camera";

const TARGET_FPS = 8;
const LOOP_MS = Math.round(1000 / TARGET_FPS);

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const inflightRef = useRef(false);

  const [mode, setMode] = useState<SourceMode>("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [roundTripMs, setRoundTripMs] = useState<number | null>(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [liveDetect, setLiveDetect] = useState(true);
  const [clientFps, setClientFps] = useState(0);
  const sentTimesRef = useRef<number[]>([]);

  const renderFrame = useCallback(
    (bitmap: CanvasImageSource, width: number, height: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(bitmap, 0, 0, width, height);
    },
    [],
  );

  const applyResult = useCallback(
    (data: DetectionResult) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const scaleX = canvas.width / data.image_width;
      const scaleY = canvas.height / data.image_height;
      drawDetections(ctx, data.detections, scaleX, scaleY);
    },
    [],
  );

  const runDetect = useCallback(
    async (blob: Blob) => {
      if (inflightRef.current) return;
      inflightRef.current = true;
      setError(null);
      const t0 = performance.now();
      try {
        const data = await detectImage(blob);
        setResult(data);
        applyResult(data);
        setRoundTripMs(performance.now() - t0);
        const now = performance.now();
        const samples = [...sentTimesRef.current, now].filter((t) => now - t < 2000);
        sentTimesRef.current = samples;
        if (samples.length > 1) {
          const span = (samples[samples.length - 1] - samples[0]) / 1000;
          setClientFps(span > 0 ? (samples.length - 1) / span : 0);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Detection failed");
      } finally {
        inflightRef.current = false;
      }
    },
    [applyResult],
  );

  const onFileChange = async (file: File | null) => {
    if (!file) return;
    setMode("file");
    setCameraOn(false);
    stopCamera();
    setLiveDetect(false);

    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = async () => {
      renderFrame(img, img.naturalWidth, img.naturalHeight);
      URL.revokeObjectURL(url);
      await runDetect(file);
    };
    img.onerror = () => {
      setError("Could not load image file");
      URL.revokeObjectURL(url);
    };
    img.src = url;
  };

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraOn(false);
    sentTimesRef.current = [];
    setClientFps(0);
  }, []);

  const captureAndDetect = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return;

    const w = video.videoWidth;
    const h = video.videoHeight;
    renderFrame(video, w, h);

    const off = document.createElement("canvas");
    off.width = w;
    off.height = h;
    off.getContext("2d")?.drawImage(video, 0, 0);
    const blob = await new Promise<Blob | null>((resolve) =>
      off.toBlob((b) => resolve(b), "image/jpeg", 0.85),
    );
    if (blob) await runDetect(blob);
  }, [renderFrame, runDetect]);

  const startCamera = async () => {
    setError(null);
    setMode("camera");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraOn(true);
    } catch {
      setError(
        "Camera permission denied or no camera found. Use file upload instead.",
      );
      setCameraOn(false);
    }
  };

  useEffect(() => () => stopCamera(), []);

  return (
    <main className="app">
      <header className="header">
        <h1>RealTime Detection</h1>
        <p className="muted">Browser demo — mock detector until ONNX model is added.</p>
      </header>

      <section className="toolbar">
        <label className="btn">
          Upload image
          <input
            type="file"
            accept="image/jpeg,image/png"
            hidden
            onChange={(e) => onFileChange(e.target.files?.[0] ?? null)}
          />
        </label>
        {!cameraOn ? (
          <button type="button" className="btn" onClick={startCamera}>
            Start camera
          </button>
        ) : (
          <>
            <button type="button" className="btn" onClick={captureAndDetect}>
              Detect frame
            </button>
            <button type="button" className="btn secondary" onClick={stopCamera}>
              Stop camera
            </button>
          </>
        )}
      </section>

      {error && <p className="error">{error}</p>}

      <div className="stage">
        <video
          ref={videoRef}
          className={mode === "camera" && cameraOn ? "preview-video" : "hidden"}
          playsInline
          muted
        />
        <canvas ref={canvasRef} className="preview-canvas" />
      </div>

      <section className="stats">
        <div>
          <span className="label">Mode</span>
          <span>{mode}</span>
        </div>
        <div>
          <span className="label">Server inference</span>
          <span>{result ? `${result.inference_ms} ms` : "—"}</span>
        </div>
        <div>
          <span className="label">Round-trip</span>
          <span>{roundTripMs != null ? `${roundTripMs.toFixed(0)} ms` : "—"}</span>
        </div>
        <div>
          <span className="label">Detections</span>
          <span>{result?.detections.length ?? "—"}</span>
        </div>
      </section>
    </main>
  );
}
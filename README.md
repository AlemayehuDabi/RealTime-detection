# RealTime-detection

Browser-based real-time object detection (YOLO on COCO classes). The **web app and API** run with a **mock detector** by default; add your quantized ONNX model locally when training/export is done.

## Project layout

- `apps/api` — FastAPI backend (`/health`, `/detect`)
- `apps/web` — Vite + React frontend (camera, upload, overlays)
- `models/` — local weights only (gitignored)
- `01_object_detection_training.ipynb` — Ultralytics training notes

## Run without a model (mock detector)

### API

Use **Python 3.11–3.14**. On **3.14**, pinned versions use prebuilt wheels for Pillow and Pydantic (older pins tried to compile from source and failed without `libjpeg` / Rust).

```bash
cd apps/api
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env   # optional; defaults to DETECTOR=mock
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Web

```bash
cd apps/web
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). The dev server proxies `/api/*` to the API on port 8000.

## Run with ONNX (after you add a model)

1. Place your quantized ONNX file under `models/` (e.g. `models/yolo26n_coco128_int8.onnx`).
2. Use `scripts/export_onnx.py` and `scripts/quantize_onnx.py` locally (not required for mock mode).
3. Set in `apps/api/.env`:

   ```env
   DETECTOR=onnx
   ONNX_MODEL_PATH=models/yolo26n_coco128_int8.onnx
   ```

4. Install ONNX runtime deps: `pip install -r requirements-onnx.txt`
5. Restart the API. The web UI stays the same; only detection quality and latency change.

## Tests

```bash
cd apps/api
source .venv/bin/activate
pytest
```
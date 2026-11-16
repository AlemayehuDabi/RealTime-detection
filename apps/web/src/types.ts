export interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  box: Box;
}

export interface DetectionResult {
  inference_ms: number;
  image_width: number;
  image_height: number;
  detections: Detection[];
}
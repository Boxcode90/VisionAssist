import cv2
import torch
import gc
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt', device='cpu', conf_threshold=0.4, input_size=320):
        """
        Optimized for mobile/CPU:
        - input_size=320 (instead of 640) -> 4x faster
        - conf_threshold=0.4 (slightly lower to catch obstacles)
        """
        self.model = YOLO(model_path)
        self.device = device
        self.model.to(device)
        self.conf_threshold = conf_threshold
        self.input_size = input_size
        self.names = self.model.names

        # Enable half-precision (FP16) for GPU speed boost
        if device == 'cuda':
            self.model.model.half()
            print("[INFO] FP16 enabled for GPU acceleration.")

    def detect(self, frame, use_tracking=False):
        h, w = frame.shape[:2]
        
        # Resize to 320x320 for speed
        frame_resized = cv2.resize(frame, (self.input_size, self.input_size))

        if use_tracking:
            results = self.model.track(frame_resized, conf=self.conf_threshold, 
                                       persist=True, verbose=False, device=self.device)
        else:
            results = self.model(frame_resized, conf=self.conf_threshold, 
                                 verbose=False, device=self.device)

        detections = []
        annotated_frame = frame.copy()

        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # Get normalized coordinates (0-1) to map back to original size
                    x1_norm, y1_norm, x2_norm, y2_norm = box.xyxyn[0].tolist()
                    x1 = int(x1_norm * w)
                    y1 = int(y1_norm * h)
                    x2 = int(x2_norm * w)
                    y2 = int(y2_norm * h)
                    
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = self.names[cls]
                    
                    track_id = None
                    if use_tracking and box.id is not None:
                        track_id = int(box.id[0])
                    
                    detections.append((x1, y1, x2, y2, conf, label, track_id))

                    if track_id is not None:
                        text = f"ID:{track_id} {label} {conf:.2f}"
                        color = (0, 255, 255)
                    else:
                        text = f"{label} {conf:.2f}"
                        color = (0, 255, 0)

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Gentle memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return detections, annotated_frame
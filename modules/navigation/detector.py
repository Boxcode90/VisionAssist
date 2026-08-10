import cv2
import torch
import gc
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt', device='cpu', conf_threshold=0.5, input_size=640):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.conf_threshold = conf_threshold
        self.input_size = input_size
        self.names = self.model.names

    def detect(self, frame, use_tracking=False):
        # Resize frame to reduce memory usage
        h, w = frame.shape[:2]
        if w != self.input_size or h != self.input_size:
            frame_resized = cv2.resize(frame, (self.input_size, self.input_size))
        else:
            frame_resized = frame

        if use_tracking:
            results = self.model.track(frame_resized, conf=self.conf_threshold, 
                                       persist=True, verbose=False)
        else:
            results = self.model(frame_resized, conf=self.conf_threshold, verbose=False)

        detections = []
        annotated_frame = frame.copy()  # we'll draw on original size later

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
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Clear GPU cache and run garbage collection every call
        torch.cuda.empty_cache()
        gc.collect()

        return detections, annotated_frame
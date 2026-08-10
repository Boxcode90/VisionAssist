import cv2
import numpy as np
import torch
import gc
from modules.navigation.detector import ObjectDetector
from modules.navigation.motion_analyzer import MotionAnalyzer

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # ========== OPTIMIZATION ==========
    # Use 'cuda' if you have RTX 2050, else 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")
    
    detector = ObjectDetector(device=device, conf_threshold=0.4, input_size=320)
    motion_analyzer = MotionAnalyzer()
    # =================================

    frame_count = 0
    detection_interval = 3  # Run YOLO every 3 frames
    detections = []         # Store last detections

    print("Milestone 3: Optimized (320px, FP16, Frame Skipping). Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Reconnecting...")
            cap.release()
            cap = cv2.VideoCapture(0)
            continue

        frame_count += 1

        # Run detection only every 3rd frame
        if frame_count % detection_interval == 0:
            detections, annotated_frame = detector.detect(frame, use_tracking=True)
        else:
            # In between, just use the last detections (Kalman filter in ByteTrack still predicts)
            # We still need to run the motion analyzer on the raw frame for optical flow
            # But we reuse the bounding boxes from the last detection.
            # For simplicity, we draw them again.
            annotated_frame = frame.copy()
            for det in detections:
                x1, y1, x2, y2, conf, label, track_id = det
                if track_id is not None:
                    color = (0, 255, 255)
                    text = f"ID:{track_id} {label}"
                else:
                    color = (0, 255, 0)
                    text = f"{label}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Analyze motion (runs every frame, but Optical Flow is cheap)
        motion_results = motion_analyzer.process(frame, detections)

        # Draw motion info on the frame
        for det in detections:
            x1, y1, x2, y2, conf, label, track_id = det
            if track_id is not None and track_id in motion_results:
                state = motion_results[track_id].get("state", "unknown")
                dx = motion_results[track_id].get("dx", 0.0)
                dy = motion_results[track_id].get("dy", 0.0)
                
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                arrow_len = int(np.sqrt(dx**2 + dy**2) * 2)
                if arrow_len > 5:
                    end_x = center_x + int(dx * 2)
                    end_y = center_y + int(dy * 2)
                    cv2.arrowedLine(annotated_frame, (center_x, center_y), (end_x, end_y), (0, 0, 255), 2)
                
                cv2.putText(annotated_frame, f"{state}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if frame_count % 30 == 0:
            tracked_info = [(d[6], d[5], motion_results.get(d[6], {}).get('state', 'N/A')) 
                            for d in detections if d[6] is not None]
            print(f"[Frame {frame_count}] Tracked: {tracked_info}")

        cv2.imshow("Optimized Detection + Motion", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Optimized test completed.")

if __name__ == "__main__":
    main()
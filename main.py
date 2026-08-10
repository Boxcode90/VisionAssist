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

    detector = ObjectDetector(device='cpu', conf_threshold=0.5)
    motion_analyzer = MotionAnalyzer()
    frame_count = 0
    print("Milestone 3: Motion Analysis enabled. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Reconnecting...")
            cap.release()
            cap = cv2.VideoCapture(0)
            continue

        frame_count += 1

        # Get detections with tracking
        detections, annotated_frame = detector.detect(frame, use_tracking=True)

        # Analyze motion
        motion_results = motion_analyzer.process(frame, detections)

        # Draw motion info on the frame
        for det in detections:
            x1, y1, x2, y2, conf, label, track_id = det
            if track_id is not None and track_id in motion_results:
                state = motion_results[track_id].get("state", "unknown")
                dx = motion_results[track_id].get("dx", 0.0)
                dy = motion_results[track_id].get("dy", 0.0)
                
                # Draw an arrow indicating motion direction
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                arrow_len = int(np.sqrt(dx**2 + dy**2) * 2)
                if arrow_len > 5:
                    end_x = center_x + int(dx * 2)
                    end_y = center_y + int(dy * 2)
                    cv2.arrowedLine(annotated_frame, (center_x, center_y), (end_x, end_y), (0, 0, 255), 2)
                
                # Show state text
                cv2.putText(annotated_frame, f"{state}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        if frame_count % 30 == 0:
            # Print motion states
            tracked_info = [(d[6], d[5], motion_results.get(d[6], {}).get('state', 'N/A')) 
                            for d in detections if d[6] is not None]
            print(f"[Frame {frame_count}] Tracked: {tracked_info}")

        cv2.imshow("Module 1 - Detection + Tracking + Motion", annotated_frame)

        # Memory cleanup
        if frame_count % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Motion analysis test completed.")

if __name__ == "__main__":
    main()
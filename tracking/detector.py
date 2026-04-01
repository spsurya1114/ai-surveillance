
from ultralytics import YOLO

class Detector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def track(self, frame):
        results = self.model.track(frame, persist=True, verbose=False)

        detections = []

        for r in results:
            if r.boxes.id is not None:
                for box, track_id in zip(r.boxes, r.boxes.id):
                    detections.append({
                        "id": int(track_id),
                        "class": int(box.cls[0]),
                        "confidence": float(box.conf[0]),
                        "bbox": box.xyxy[0].tolist()
                    })

        return detections


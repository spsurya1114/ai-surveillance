
from reid.reid_model import ReID
from reid.matcher import Matcher
import cv2
import os
from tracking.detector import Detector
from tracking.logger import EventLogger
from anomaly.detector import AnomalyDetector


def process_video(path, camera_id):
    cap = cv2.VideoCapture(path)

    detector = Detector()
    reid = ReID()
    matcher = Matcher()
    anomaly_detector = AnomalyDetector(threshold_time=5)

    logger = EventLogger(save_path=f"data/logs/{camera_id}.json")

    # Create output folder
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_id = 0
    print("VIDEO PATH:", path)
    print("CAP OPEN:", cap.isOpened())


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.track(frame)
        print("RAW DETECTIONS:", detections)
        for i, d in enumerate(detections):
            d["id"] = i
        print("WITH IDS:", detections)
        frame_width = frame.shape[1]

        # -----------------------------
        # Re-ID
        # -----------------------------
        for d in detections:
            x1, y1, x2, y2 = map(int, d["bbox"])

            person_img = frame[y1:y2, x1:x2]

            if person_img.size == 0:
                continue

            features = reid.extract_features(person_img)
            global_id = matcher.match(features)

            d["global_id"] = global_id
        print("DETECTIONS:", detections)

        # -----------------------------
        # Logging
        # -----------------------------
        new_events = logger.update(frame_id, fps, detections, frame_width, camera_id)
        print("NEW EVENTS:", new_events)
        # -----------------------------
        # Anomaly detection
        # -----------------------------
        for event in new_events:
            anomalies = anomaly_detector.check(event)

            for a in anomalies:
                print("⚠️ ANOMALY DETECTED:", a)

        # -----------------------------
        # Draw boxes
        # -----------------------------
        for d in detections:
            x1, y1, x2, y2 = map(int, d["bbox"])
            gid = d.get("global_id", -1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"GID: {gid}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            

        # -----------------------------
        # Save frame (NOT every frame)
        # -----------------------------
        if frame_id % 30 == 0:  # save every 30 frames (~1 sec)
            cv2.imwrite(f"{output_dir}/{camera_id}_frame_{frame_id}.jpg", frame)

        frame_id += 1

    cap.release()
    logger.save()

    print(f"{camera_id} processing completed.")

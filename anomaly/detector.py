
import math
import json
import os
import cv2
import numpy as np
from collections import deque

def get_bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


class LoiteringDetector:
    def __init__(self, threshold):
        self.threshold = threshold
        self.first_seen = {}  # person_id -> timestamp
        self.triggered = set()  # person_id

    def update(self, person_id, timestamp, camera_id):
        anomalies = []
        if person_id not in self.first_seen:
            self.first_seen[person_id] = timestamp
            return anomalies

        duration = timestamp - self.first_seen[person_id]
        if duration > self.threshold and person_id not in self.triggered:
            self.triggered.add(person_id)
            anomalies.append({
                "anomaly_type": "loitering",
                "severity": "MEDIUM",
                "risk_score": 25,
                "timestamp": timestamp,
                "camera_id": camera_id,
                "person_id": person_id,
                "description": f"Person {person_id} loitering for {round(duration, 2)} seconds."
            })
        return anomalies

    def remove_person(self, person_id):
        if person_id in self.first_seen:
            del self.first_seen[person_id]
        self.triggered.discard(person_id)


class RestrictedZoneDetector:
    def __init__(self, zones_config):
        self.zones = zones_config or []
        self.triggered = set()  # (person_id, zone_name)

    def update(self, person_id, bbox, confidence, timestamp, camera_id):
        anomalies = []
        cx, cy = get_bbox_center(bbox)

        for zone in self.zones:
            zone_name = zone["name"]
            polygon = np.array(zone["polygon"], dtype=np.int32)

            dist = cv2.pointPolygonTest(polygon, (cx, cy), False)
            if dist >= 0:
                key = (person_id, zone_name)
                if key not in self.triggered:
                    self.triggered.add(key)
                    anomalies.append({
                        "anomaly_type": "intrusion",
                        "severity": "HIGH",
                        "risk_score": 40,
                        "timestamp": timestamp,
                        "camera_id": camera_id,
                        "person_id": person_id,
                        "confidence": confidence,
                        "zone": zone_name,
                        "description": f"Person {person_id} intruded restricted zone '{zone_name}'."
                    })
            else:
                key = (person_id, zone_name)
                self.triggered.discard(key)

        return anomalies

    def remove_person(self, person_id):
        keys_to_remove = [k for k in self.triggered if k[0] == person_id]
        for k in keys_to_remove:
            self.triggered.remove(k)


class DirectionDetector:
    def __init__(self, allowed_direction):
        self.allowed = allowed_direction
        self.trajectories = {}  # person_id -> deque of (cx, cy, timestamp)
        self.triggered = set()  # person_id

    def update(self, person_id, bbox, timestamp, camera_id):
        anomalies = []
        if not self.allowed:
            return anomalies

        cx, cy = get_bbox_center(bbox)
        if person_id not in self.trajectories:
            self.trajectories[person_id] = deque(maxlen=15)

        self.trajectories[person_id].append((cx, cy, timestamp))

        traj = self.trajectories[person_id]
        if len(traj) >= 5:
            x_first, y_first, _ = traj[0]
            x_last, y_last, _ = traj[-1]

            dx = x_last - x_first
            dy = y_last - y_first

            dist = math.hypot(dx, dy)
            if dist > 30:  # evaluate only if significant movement
                move_angle = math.degrees(math.atan2(dy, dx)) % 360
                allowed_angle = self.allowed["angle"]
                tolerance = self.allowed["tolerance"]

                diff = abs(move_angle - allowed_angle) % 360
                if diff > 180:
                    diff = 360 - diff

                if diff > tolerance:
                    if person_id not in self.triggered:
                        self.triggered.add(person_id)
                        anomalies.append({
                            "anomaly_type": "wrong_direction",
                            "severity": "MEDIUM",
                            "risk_score": 20,
                            "timestamp": timestamp,
                            "camera_id": camera_id,
                            "person_id": person_id,
                            "description": f"Person {person_id} moving in wrong direction (angle {round(move_angle, 1)} deg, allowed {allowed_angle} deg)."
                        })
                else:
                    self.triggered.discard(person_id)
        return anomalies

    def remove_person(self, person_id):
        if person_id in self.trajectories:
            del self.trajectories[person_id]
        self.triggered.discard(person_id)


class VelocityDetector:
    def __init__(self, threshold):
        self.threshold = threshold
        self.last_pos = {}  # person_id -> (cx, cy, timestamp)
        self.triggered = set()  # person_id

    def update(self, person_id, bbox, timestamp, camera_id):
        anomalies = []
        cx, cy = get_bbox_center(bbox)

        if person_id not in self.last_pos:
            self.last_pos[person_id] = (cx, cy, timestamp)
            return anomalies

        lx, ly, lt = self.last_pos[person_id]
        dt = timestamp - lt

        if dt > 0.05:
            dist = math.hypot(cx - lx, cy - ly)
            velocity = dist / dt

            if velocity > self.threshold and person_id not in self.triggered:
                self.triggered.add(person_id)
                anomalies.append({
                    "anomaly_type": "sudden_movement",
                    "severity": "MEDIUM",
                    "risk_score": 20,
                    "timestamp": timestamp,
                    "camera_id": camera_id,
                    "person_id": person_id,
                    "description": f"Person {person_id} abnormal movement/running detected (velocity {round(velocity, 1)} px/s)."
                })
            elif velocity <= self.threshold * 0.7:
                self.triggered.discard(person_id)

            self.last_pos[person_id] = (cx, cy, timestamp)

        return anomalies

    def remove_person(self, person_id):
        if person_id in self.last_pos:
            del self.last_pos[person_id]
        self.triggered.discard(person_id)


class CrowdDetector:
    def __init__(self, threshold):
        self.threshold = threshold
        self.is_crowded = False

    def update(self, people_count, timestamp, camera_id):
        anomalies = []
        if people_count > self.threshold:
            if not self.is_crowded:
                self.is_crowded = True
                anomalies.append({
                    "anomaly_type": "crowd_anomaly",
                    "severity": "LOW",
                    "risk_score": 15,
                    "timestamp": timestamp,
                    "camera_id": camera_id,
                    "description": f"Crowd detected: {people_count} people in scene (threshold {self.threshold})."
                })
        else:
            self.is_crowded = False
        return anomalies


class AbandonedObjectDetector:
    def __init__(self, threshold):
        self.threshold = threshold
        self.objects = {}  # object_id -> {"first_seen": ts, "last_pos": (cx, cy), "stationary_since": ts}
        self.triggered = set()

    def update(self, object_id, bbox, timestamp, camera_id, people_centers):
        anomalies = []
        cx, cy = get_bbox_center(bbox)

        if object_id not in self.objects:
            self.objects[object_id] = {
                "first_seen": timestamp,
                "last_pos": (cx, cy),
                "stationary_since": timestamp
            }
            return anomalies

        obj_state = self.objects[object_id]
        lx, ly = obj_state["last_pos"]

        dist = math.hypot(cx - lx, cy - ly)
        if dist > 10:  # object moved
            obj_state["last_pos"] = (cx, cy)
            obj_state["stationary_since"] = timestamp
            self.triggered.discard(object_id)
        else:
            stationary_duration = timestamp - obj_state["stationary_since"]
            if stationary_duration > self.threshold and object_id not in self.triggered:
                # Check owner proximity
                owner_nearby = False
                for px, py in people_centers:
                    if math.hypot(cx - px, cy - py) < 150:
                        owner_nearby = True
                        break

                if not owner_nearby:
                    self.triggered.add(object_id)
                    anomalies.append({
                        "anomaly_type": "abandoned_object",
                        "severity": "MEDIUM",
                        "risk_score": 30,
                        "timestamp": timestamp,
                        "camera_id": camera_id,
                        "object_id": object_id,
                        "description": f"Abandoned object (ID {object_id}) stationary for {round(stationary_duration, 1)} seconds."
                    })
        return anomalies

    def remove_object(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        self.triggered.discard(object_id)


class AnomalyDetector:
    def __init__(self, threshold_time=10, config_path="config.json"):
        # Backward compatibility fields
        self.entry_times = {}
        self.threshold = threshold_time

        # Load config
        self.config = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    self.config = json.load(f)
            except Exception as e:
                print(f"Failed to load config.json: {e}")

        # Override backward compatibility threshold if config exists
        self.threshold = self.config.get("loitering_threshold", self.threshold)

        # Instantiate sub-detectors
        self.loitering_detector = LoiteringDetector(self.threshold)
        self.restricted_detector = RestrictedZoneDetector(self.config.get("restricted_zones", {}).get("CAM_01", []))
        self.direction_detector = DirectionDetector(None)
        self.velocity_detector = VelocityDetector(self.config.get("velocity_threshold", 100.0))
        self.crowd_detector = CrowdDetector(self.config.get("crowd_threshold", 3))
        self.abandoned_detector = AbandonedObjectDetector(self.config.get("abandoned_object_threshold", 10.0))

        self.active_people_ids = set()
        self.active_object_ids = set()
        self.current_camera_id = None

    def set_camera(self, camera_id):
        """Update sub-detectors based on the camera configuration."""
        if self.current_camera_id == camera_id:
            return
        self.current_camera_id = camera_id

        zones = self.config.get("restricted_zones", {}).get(camera_id, [])
        self.restricted_detector = RestrictedZoneDetector(zones)

        allowed_dir = self.config.get("allowed_directions", {}).get(camera_id, None)
        self.direction_detector = DirectionDetector(allowed_dir)

    def update(self, frame_id, fps, detections, camera_id, frame=None):
        timestamp = frame_id / fps
        self.set_camera(camera_id)

        # Distinguish people (class 0) from potential abandoned objects (class 24, 25, 26, 28)
        people_dets = []
        object_dets = []

        # YOLO classes: 0 = person, 24 = backpack, 25 = umbrella, 26 = handbag, 28 = suitcase
        OBJECT_CLASSES = {24, 25, 26, 28}

        for d in detections:
            cls_id = d.get("class", 0)
            if cls_id == 0:
                people_dets.append(d)
            elif cls_id in OBJECT_CLASSES:
                object_dets.append(d)

        current_people_ids = set(d.get("global_id", d["id"]) for d in people_dets)
        current_object_ids = set(d["id"] for d in object_dets)

        # Clean up stale tracks
        for pid in self.active_people_ids - current_people_ids:
            self.loitering_detector.remove_person(pid)
            self.restricted_detector.remove_person(pid)
            self.direction_detector.remove_person(pid)
            self.velocity_detector.remove_person(pid)

        for oid in self.active_object_ids - current_object_ids:
            self.abandoned_detector.remove_object(oid)

        self.active_people_ids = current_people_ids
        self.active_object_ids = current_object_ids

        anomalies = []

        # Get people centers for proximity checks
        people_centers = []
        for d in people_dets:
            cx, cy = get_bbox_center(d["bbox"])
            people_centers.append((cx, cy))

        # Check person-based anomalies
        for d in people_dets:
            pid = d.get("global_id", d["id"])
            bbox = d["bbox"]
            conf = d.get("confidence", 1.0)

            # Loitering
            anomalies.extend(self.loitering_detector.update(pid, timestamp, camera_id))
            # Intrusion
            anomalies.extend(self.restricted_detector.update(pid, bbox, conf, timestamp, camera_id))
            # Direction
            anomalies.extend(self.direction_detector.update(pid, bbox, timestamp, camera_id))
            # Velocity
            anomalies.extend(self.velocity_detector.update(pid, bbox, timestamp, camera_id))

        # Crowd anomalies
        anomalies.extend(self.crowd_detector.update(len(people_dets), timestamp, camera_id))

        # Abandoned objects
        for d in object_dets:
            oid = d["id"]
            bbox = d["bbox"]
            anomalies.extend(self.abandoned_detector.update(oid, bbox, timestamp, camera_id, people_centers))

        # Assign default risk settings (if not already set in sub-detector)
        weights = self.config.get("risk_weights", {})
        for a in anomalies:
            atype = a["anomaly_type"]
            a["risk_score"] = weights.get(atype, a.get("risk_score", 20))
            # Add severity classification
            if a["risk_score"] >= 90:
                a["severity"] = "CRITICAL"
            elif a["risk_score"] >= 70:
                a["severity"] = "HIGH"
            elif a["risk_score"] >= 40:
                a["severity"] = "MEDIUM"
            else:
                a["severity"] = "LOW"

        return anomalies

    def check(self, event):
        """Backward compatibility for transition events (entry/exit)."""
        anomalies = []
        pid = event["person_id"]
        event_type = event["event_type"]
        timestamp = event["timestamp"]

        # ENTRY → record time
        if event_type == "entry":
            self.entry_times[pid] = timestamp

        # EXIT → check duration
        elif event_type == "exit":
            if pid in self.entry_times:
                duration = timestamp - self.entry_times[pid]

                if duration > self.threshold:
                    anomalies.append({
                        "anomaly_type": "loitering",
                        "severity": "MEDIUM",
                        "risk_score": 25,
                        "person_id": pid,
                        "duration": round(duration, 2),
                        "camera_id": event["camera_id"],
                        "timestamp": timestamp,
                        "description": f"Person {pid} loitering for {round(duration, 2)} seconds."
                    })

                del self.entry_times[pid]

        return anomalies
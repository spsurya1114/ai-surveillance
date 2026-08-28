
import json
import os
import uuid
from tracking.zone import get_zone


class EventLogger:
    def __init__(self, save_path="data/logs/events.json", config_path="config.json"):
        self.save_path = save_path
        self.events = []
        self.active_ids = set()
        self.lost_ids = {}  # Map YOLO track id -> frames missing count
        self.last_zone = {}
        self.global_id_cache = {}  # Map YOLO track id -> Re-ID global_id

        # Load config to get tracking settings
        self.config = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    self.config = json.load(f)
            except Exception as e:
                print(f"Failed to load config in EventLogger: {e}")

        # Load existing events if they exist to keep compatibility
        if os.path.exists(save_path):
            try:
                with open(save_path, "r") as f:
                    self.events = json.load(f)
            except:
                self.events = []

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def update(self, frame_id, fps, detections, frame_width, camera_id="CAM_01"):
        timestamp = frame_id / fps
        new_events = []

        # Update cache mapping for global IDs
        for d in detections:
            track_id = d["id"]
            gid = d.get("global_id", track_id)
            self.global_id_cache[track_id] = gid

        current_ids = set([d["id"] for d in detections])

        # Assign zones
        for d in detections:
            d["zone"] = get_zone(d["bbox"], frame_width)

        # Retrieve max_missing_frames from config
        max_missing = self.config.get("tracking", {}).get("max_missing_frames", 30)

        # 1. PROCESS ENTRIES & REAPPEARANCES
        for d in detections:
            pid = d["id"]
            gid = self.global_id_cache.get(pid, pid)
            zone = d["zone"]

            if pid in self.lost_ids:
                # Reappeared within grace period: restore state, do not trigger entry
                del self.lost_ids[pid]
                self.active_ids.add(pid)
            elif pid not in self.active_ids:
                # Brand new entry
                event = {
                    "event_id": str(uuid.uuid4()),
                    "timestamp": round(timestamp, 2),
                    "camera_id": camera_id,
                    "person_id": gid,
                    "object_id": None,
                    "event_type": "entry",
                    "severity": "LOW",
                    "risk_score": 0,
                    "confidence": round(d.get("confidence", 1.0), 2),
                    "zone": zone,
                    "description": f"Person {gid} entered scene at zone {zone}.",
                    "evidence_path": None,
                    "from_zone": None,
                    "to_zone": zone
                }
                self.events.append(event)
                new_events.append(event)
                self.active_ids.add(pid)
                self.last_zone[pid] = zone

        # 2. PROCESS MOVEMENTS (only for active tracks)
        for d in detections:
            pid = d["id"]
            current_zone = d["zone"]
            gid = self.global_id_cache.get(pid, pid)

            if pid in self.last_zone and self.last_zone[pid] != current_zone:
                event = {
                    "event_id": str(uuid.uuid4()),
                    "timestamp": round(timestamp, 2),
                    "camera_id": camera_id,
                    "person_id": gid,
                    "object_id": None,
                    "event_type": "movement",
                    "severity": "LOW",
                    "risk_score": 0,
                    "confidence": round(d.get("confidence", 1.0), 2),
                    "zone": current_zone,
                    "description": f"Person {gid} moved from {self.last_zone[pid]} to {current_zone}.",
                    "evidence_path": None,
                    "from_zone": self.last_zone[pid],
                    "to_zone": current_zone
                }
                self.events.append(event)
                new_events.append(event)
                self.last_zone[pid] = current_zone

        # 3. PROCESS TRACK LOSS AND PENDING EXITS
        # For active IDs that disappeared in this frame:
        for pid in list(self.active_ids):
            if pid not in current_ids:
                self.active_ids.remove(pid)
                self.lost_ids[pid] = 0

        # Increment missing count for all lost IDs and check for exits
        for pid in list(self.lost_ids.keys()):
            if pid not in current_ids:
                self.lost_ids[pid] += 1
                if self.lost_ids[pid] > max_missing:
                    # Grace period expired, trigger exit event
                    gid = self.global_id_cache.get(pid, pid)
                    event = {
                        "event_id": str(uuid.uuid4()),
                        "timestamp": round(timestamp, 2),
                        "camera_id": camera_id,
                        "person_id": gid,
                        "object_id": None,
                        "event_type": "exit",
                        "severity": "LOW",
                        "risk_score": 0,
                        "confidence": 1.0,
                        "zone": self.last_zone.get(pid, None),
                        "description": f"Person {gid} exited scene from zone {self.last_zone.get(pid, None)}.",
                        "evidence_path": None,
                        "from_zone": self.last_zone.get(pid, None),
                        "to_zone": None
                    }
                    self.events.append(event)
                    new_events.append(event)

                    # Cleanup state for this person
                    del self.lost_ids[pid]
                    if pid in self.last_zone:
                        del self.last_zone[pid]
                    if pid in self.global_id_cache:
                        del self.global_id_cache[pid]

        return new_events

    def log_anomaly(self, anomaly, risk_score, risk_level, evidence_path=None, evidence_clip_path=None):
        """Logs an anomaly event into the unified log with appropriate schema."""
        event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": round(anomaly.get("timestamp", 0.0), 2),
            "camera_id": anomaly["camera_id"],
            "person_id": anomaly.get("person_id"),
            "object_id": anomaly.get("object_id"),
            "event_type": "anomaly",
            "anomaly_type": anomaly["anomaly_type"],
            "severity": risk_level,
            "risk_score": risk_score,
            "confidence": round(anomaly.get("confidence", 1.0), 2),
            "zone": anomaly.get("zone"),
            "description": anomaly["description"],
            "evidence_path": evidence_path,
            "evidence_clip_path": evidence_clip_path
        }
        self.events.append(event)
        return event

    def save(self):
        try:
            with open(self.save_path, "w") as f:
                json.dump(self.events, f, indent=2)
        except Exception as e:
            print(f"Failed to save events to {self.save_path}: {e}")



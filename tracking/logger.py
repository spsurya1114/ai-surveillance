
import json
import os
from tracking.zone import get_zone


class EventLogger:
    def __init__(self, save_path="data/logs/events.json"):
        self.save_path = save_path
        self.events = []
        self.active_ids = set()
        self.last_zone = {}

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def update(self, frame_id, fps, detections, frame_width, camera_id="CAM_01"):
        timestamp = frame_id / fps
        new_events = []

        current_ids = set([d["id"] for d in detections])

        # Assign zones
        for d in detections:
            d["zone"] = get_zone(d["bbox"], frame_width)

        # ENTRY
        for pid in current_ids - self.active_ids:
            det = next(d for d in detections if d["id"] == pid)

            event = {
                "event_type": "entry",
                "person_id": det.get("global_id", pid),
                "camera_id": camera_id,
                "timestamp": round(timestamp, 2),
                "from_zone": None,
                "to_zone": det["zone"]
            }

            self.events.append(event)
            new_events.append(event)
            self.last_zone[pid] = det["zone"]

        # MOVEMENT
        for d in detections:
            pid = d["id"]
            current_zone = d["zone"]

            if pid in self.last_zone and self.last_zone[pid] != current_zone:
                event = {
                    "event_type": "movement",
                    "person_id": d.get("global_id", pid),
                    "from_zone": self.last_zone[pid],
                    "to_zone": current_zone,
                    "timestamp": round(timestamp, 2),
                    "camera_id": camera_id
                }

                self.events.append(event)
                new_events.append(event)
                self.last_zone[pid] = current_zone

        # EXIT
        for pid in self.active_ids - current_ids:
            event = {
                "event_type": "exit",
                "person_id": pid,
                "from_zone": self.last_zone.get(pid, None),
                "to_zone": None,
                "timestamp": round(timestamp, 2),
                "camera_id": camera_id
            }

            self.events.append(event)
            new_events.append(event)

            if pid in self.last_zone:
                del self.last_zone[pid]

        self.active_ids = current_ids

        return new_events  # VERY IMPORTANT

    def save(self):
        with open(self.save_path, "w") as f:
            json.dump(self.events, f, indent=2)


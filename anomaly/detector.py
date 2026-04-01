
class AnomalyDetector:
    def __init__(self, threshold_time=10):
        self.entry_times = {}
        self.threshold = threshold_time

    def check(self, event):
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
                        "type": "loitering",
                        "person_id": pid,
                        "duration": round(duration, 2),
                        "camera_id": event["camera_id"]
                    })

                del self.entry_times[pid]

        return anomalies